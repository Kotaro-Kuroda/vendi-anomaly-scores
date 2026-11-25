"""Streamlit review of image datasets using Vendi Anomaly Scores."""

from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2

from methods.vas import VAS

PAGE_SIZE = 24
PREPROCESS = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((672, 672), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@dataclass(frozen=True)
class ReviewImage:
    path: str
    label: str | None
    thumbnail: bytes


@st.cache_resource
def load_model(device):
    return torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14_reg", trust_repo=True
    ).to(device).eval()


def read_image(upload):
    # Streamlit includes the selected directory name in each upload path.
    parts = PurePosixPath(upload.name).parts
    relative = parts[1:] if len(parts) > 1 else parts
    label = relative[0] if len(relative) > 1 else None
    with Image.open(BytesIO(upload.getvalue())) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    thumbnail = image.copy()
    thumbnail.thumbnail((320, 320))
    buffer = BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=85)
    return ReviewImage("/".join(relative), label, buffer.getvalue()), image


def rank_images(images, features, indices, device):
    if len(indices) < 2:
        return []
    selected_features = features[indices]
    vas = VAS(device=device).fit(selected_features)
    scores = vas.decision_function(selected_features)
    if not np.isfinite(scores).all():
        raise ValueError("有限の VAS スコアを計算できませんでした。")
    ranked = [(images[index], float(score)) for index, score in zip(indices, scores)]
    return sorted(ranked, key=lambda entry: (-entry[1], entry[0].path))


@torch.inference_mode()
def analyze(uploads, progress):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4 if device == "cuda" else 2
    model = None
    images, errors, embeddings, batch = [], [], [], []

    for index, upload in enumerate(uploads, start=1):
        try:
            record, image = read_image(upload)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            errors.append(upload.name)
        else:
            images.append(record)
            batch.append(PREPROCESS(image))

        if len(batch) == batch_size or (batch and len(images) >= 2 and index == len(uploads)):
            if model is None:
                progress.progress((index - 1) / len(uploads), text="DINOv2 を準備中")
                model = load_model(device)
            embeddings.append(model(torch.stack(batch).to(device)).float().cpu())
            batch.clear()
        progress.progress(index / len(uploads), text=f"画像を処理中 {index} / {len(uploads)}")

    if len(images) < 2:
        return len(images), [], {}, errors

    progress.progress(1.0, text="VAS スコアを計算中")
    features = torch.cat(embeddings).numpy()
    by_label = defaultdict(list)
    for index, record in enumerate(images):
        if record.label is not None:
            by_label[record.label].append(index)

    overall = rank_images(images, features, list(range(len(images))), device)
    classes = {
        label: rank_images(images, features, indices, device)
        for label, indices in sorted(by_label.items())
    }
    return len(images), overall, classes, errors


def show_ranking(ranked, view):
    pages = (len(ranked) + PAGE_SIZE - 1) // PAGE_SIZE
    page = st.number_input(
        "ページ", min_value=1, max_value=pages, step=1,
        key=f"review_page_{view}", disabled=pages == 1,
    )
    start = (page - 1) * PAGE_SIZE
    visible = ranked[start : start + PAGE_SIZE]
    st.caption(f"{start + 1}–{start + len(visible)} / {len(ranked)} 枚")
    for offset in range(0, len(visible), 4):
        columns = st.columns(4)
        for position, (record, score) in enumerate(visible[offset : offset + 4]):
            with columns[position]:
                st.image(record.thumbnail, width="stretch")
                st.markdown(f"**{start + offset + position + 1} 位** · VAS {score:.6g}")
                st.text(record.path)


def main():
    st.set_page_config(page_title="画像データセット点検", layout="wide")
    st.title("画像データセット点検")
    st.write("VAS が高い画像から順に目視で確認できます。")
    uploads = st.file_uploader(
        "画像フォルダ",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files="directory",
        help="ラベルを確認する場合は、クラス別サブフォルダを含む親フォルダを選んでください。",
    )
    upload_ids = tuple(upload.file_id for upload in uploads)
    if st.session_state.get("review_uploads") != upload_ids:
        st.session_state.review_uploads = upload_ids
        st.session_state.pop("review_result", None)

    if st.button("解析", type="primary", disabled=not uploads):
        st.session_state.pop("review_result", None)
        progress = st.progress(0, text="画像を準備中")
        try:
            st.session_state.review_result = analyze(uploads, progress)
        except Exception as error:
            st.error(f"解析に失敗しました。再試行できます。\n\n{type(error).__name__}: {error}")
        finally:
            progress.empty()

    result = st.session_state.get("review_result")
    if result is None:
        return
    count, overall, classes, errors = result
    if errors:
        st.warning(f"読み込めなかった画像 {len(errors)} 枚をスキップしました。")
        with st.expander("読み込めなかった画像"):
            st.text("\n".join(errors))
    if count < 2:
        st.info("スコアの計算には有効な画像が 2 枚以上必要です。")
        return

    view = st.radio(
        "ランキング", ["全体の外れ値", "ラベル付けミス候補"], horizontal=True
    )
    if view == "全体の外れ値":
        show_ranking(overall, "overall")
    elif classes:
        label = st.selectbox("クラス", list(classes))
        if classes[label]:
            st.caption("同じクラスの中で見た目が異なる候補を表示しています。")
            show_ranking(classes[label], f"class_{label}")
        else:
            st.info("クラス内のスコアの計算には有効な画像が 2 枚以上必要です。")
    else:
        st.info("クラス別サブフォルダを含む親フォルダを選んでください。")


if __name__ == "__main__":
    main()
