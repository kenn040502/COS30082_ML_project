import streamlit as st
from PIL import Image
import torch
import os
import sys
import numpy as np
import io
import traceback
from pathlib import Path
import tempfile

# Determine repo root reliably (two levels up from gui file)
ROOT = Path(__file__).resolve().parents[1]
APP3 = ROOT / 'Approach 3'
if str(APP3) not in sys.path:
    sys.path.insert(0, str(APP3))

try:
    from approach3 import eval_core
    load_model_for_eval = eval_core.load_model_for_eval
except Exception:
    load_model_for_eval = None

st.set_page_config(page_title='Herbarium — Demo', layout='wide')

st.markdown('# Herbarium — Demo')

left, right = st.columns([2, 1])

with left:
    st.subheader('Upload Image')
    uploaded = st.file_uploader('Image upload', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    if uploaded is None:
        st.info('Drop an image or click to upload (jpg/png)')
    else:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, use_column_width=True)
    st.markdown('---')
    st.subheader('Sample Image of Predicted Taxon')
    sample_slot = st.empty()

with right:
    st.subheader('Controls')
    approach = st.selectbox('Approach', ['Approach 3', 'Approach 2', 'Approach 1'])
    data_root = st.text_input('Data root', value=str(ROOT / 'AML_project_herbarium_dataset'))
    top_k = st.slider('Top-K', 1, 10, 5)
    use_gpu = st.checkbox('Use GPU if available', value=False)
    # Choose device
    device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')

    # file upload helpers for ckpt/classifier
    if approach == 'Approach 3':
        ckpt_path_input = st.text_input('Approach 3 checkpoint (.pt)', '')
        ckpt_upload = st.file_uploader('Or upload Approach 3 .pt', type=['pt'])
    elif approach == 'Approach 2':
        extractor_path_input = st.text_input('Extractor weights (optional)', '')
        extractor_upload = st.file_uploader('Or upload extractor weights', type=['pt', 'pth'], key='ext_up')
        classifier_path_input = st.text_input('Classifier (.pkl)', value=str(ROOT / 'Approach 2' / 'Approach2_v1' / 'weights' / 'sklearn_model.pkl'))
        classifier_upload = st.file_uploader('Or upload classifier .pkl', type=['pkl'], key='clf_up')
    else:
        ckpt_path_input = st.text_input('Approach 1 checkpoint (.pt)', '')
        ckpt_upload = st.file_uploader('Or upload Approach 1 .pt', type=['pt'], key='a1_up')
    # container for sample gallery
    sample_container = st.container()

    st.markdown('---')
    rank = st.selectbox('Taxonomic rank', ['Species'])
    submit = st.button('Submit', use_container_width=True)

    st.markdown('---')
    st.subheader('Predictions')
    pred_box = st.empty()


@st.cache_resource
def try_load_approach3(ckpt_path, data_root, device=torch.device('cpu')):
    if load_model_for_eval is None:
        raise RuntimeError('Approach 3 helper not found in repo')
    model, preprocess, meta = load_model_for_eval(ckpt_path, data_root, device=device)
    return model, preprocess, meta


def _write_uploaded_tmp(uploaded_file):
    if uploaded_file is None:
        return ''
    suffix = Path(uploaded_file.name).suffix
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded_file.getbuffer())
    tf.flush()
    tf.close()
    return tf.name


def run_model_inference(approach, img_pil, top_k, device=torch.device('cpu'), **kwargs):
    # Approach 3
    if approach == 'Approach 3':
        ckpt = kwargs.get('ckpt')
        # allow uploaded ckpt
        if kwargs.get('ckpt_upload'):
            ckpt = _write_uploaded_tmp(kwargs.get('ckpt_upload'))
        model_obj, preprocess, meta = try_load_approach3(ckpt, kwargs.get('data_root'), device=device)
        backbone, proj, clf = model_obj.backbone, model_obj.proj, model_obj.clf
        tensor = preprocess(img_pil).unsqueeze(0).to(device)
        backbone.to(device); proj.to(device); clf.to(device)
        backbone.eval(); proj.eval(); clf.eval()
        with torch.no_grad():
            h = backbone(tensor)
            if h.dim() > 2:
                h = h.mean(dim=tuple(range(2, h.dim())))
            z = proj(h)
            logits = clf(z)
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        classes = meta.get('classes', [str(i) for i in range(len(probs))])
        idxs = probs.argsort()[::-1][:top_k]
        return [(classes[int(i)], float(probs[int(i)])) for i in idxs]

    # Approach 1: use cdna_pipeline loader
    if approach == 'Approach 1':
        APP1 = ROOT / 'Approach 1' / 'cdna_pipeline'
        if str(APP1) not in sys.path:
            sys.path.insert(0, str(APP1))
        try:
            from models.feature_extractor import get_backbone
            from models.classifier import ClassifierHead
            from utils.transforms import get_transforms
        except Exception as e:
            raise RuntimeError(f'Approach 1 modules not importable: {e}')
        ckpt = kwargs.get('ckpt')
        if kwargs.get('ckpt_upload'):
            ckpt = _write_uploaded_tmp(kwargs.get('ckpt_upload'))
        if not ckpt or not os.path.exists(ckpt):
            raise RuntimeError('Approach 1 checkpoint required')
        ck = torch.load(ckpt, map_location=device)
        F, feat_dim = get_backbone(ck.get('backbone_name','dinov2'))
        C = ClassifierHead(feat_dim, ck.get('num_classes', 100))
        F.load_state_dict(ck['feature_extractor_state_dict'])
        C.load_state_dict(ck['classifier_state_dict'])
        preprocess = get_transforms(train=False, backbone='dinov2')
        tensor = preprocess(img_pil).unsqueeze(0).to(device)
        F.to(device); C.to(device)
        F.eval(); C.eval()
        with torch.no_grad():
            feats = F(tensor)
            if feats.dim() > 2:
                feats = feats.mean(dim=tuple(range(2, feats.dim())))
            logits = C(feats)
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        classes = ck.get('class_ids', [str(i) for i in range(len(probs))])
        idxs = probs.argsort()[::-1][:top_k]
        return [(classes[int(i)], float(probs[int(i)])) for i in idxs]

    if approach == 'Approach 2':
        # load classifier
        import joblib
        clf_path = kwargs.get('classifier')
        # handle uploaded classifier
        if kwargs.get('classifier_upload'):
            clf_path = _write_uploaded_tmp(kwargs.get('classifier_upload'))
        if not os.path.exists(clf_path):
            raise RuntimeError('Classifier file not found')
        # cached load
        pack = _cached_load_classifier(clf_path)
        clf = pack.get('model', pack)
        # load extractor (try local extractor, otherwise fallback to Approach3 dino)
        APP2_SRC = ROOT / 'Approach 2' / 'Approach2_v1' / 'src'
        if str(APP2_SRC) not in sys.path:
            sys.path.insert(0, str(APP2_SRC))
        try:
            from extractor_dinov2 import load_dinov2_feature_extractor, get_transform
        except Exception:
            from dino_model import load_dino as load_dinov2_feature_extractor
            def get_transform(sz=518):
                from torchvision import transforms
                return transforms.Compose([
                    transforms.Resize((sz, sz)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
                ])
        extractor = kwargs.get('extractor', '')
        # uploaded extractor
        if kwargs.get('extractor_upload'):
            extractor = _write_uploaded_tmp(kwargs.get('extractor_upload'))
        feat_model = _cached_load_extractor(extractor or kwargs.get('ckpt',''), device)
        preprocess = get_transform(518)
        tensor = preprocess(img_pil).unsqueeze(0).to(device)
        feat_model.to(device)
        feat_model.eval()
        with torch.no_grad():
            f = feat_model(tensor)
            if f.ndim > 2:
                f = torch.flatten(f, 1)
            X = f.cpu().numpy()
        import numpy as _np
        if hasattr(clf, 'predict_proba'):
            P = clf.predict_proba(X)
        else:
            S = clf.decision_function(X)
            if S.ndim == 1:
                S = _np.stack([-S, S], axis=1)
            E = _np.exp(S - S.max(axis=1, keepdims=True))
            P = E / E.sum(axis=1, keepdims=True)
        probs = P[0]
        if 'classes' in pack:
            classes = list(pack['classes'])
        elif hasattr(clf, 'classes_'):
            classes = [int(x) if isinstance(x, (int, str)) and str(x).isdigit() else x for x in clf.classes_]
        else:
            classes = [str(i) for i in range(len(probs))]
        idxs = probs.argsort()[::-1][:top_k]
        return [(classes[int(i)], float(probs[int(i)])) for i in idxs]


@st.cache_resource
def _cached_load_classifier(clf_path):
    import joblib
    pack = joblib.load(clf_path)
    return pack


@st.cache_resource
def _cached_load_extractor(extractor_path, device=torch.device('cpu')):
    # import the fallback loader in the same way the main code does
    APP2_SRC = ROOT / 'Approach 2' / 'Approach2_v1' / 'src'
    if str(APP2_SRC) not in sys.path:
        sys.path.insert(0, str(APP2_SRC))
    try:
        from extractor_dinov2 import load_dinov2_feature_extractor, get_transform
        feat_model, meta = load_dinov2_feature_extractor(extractor_path or '', device)
        return feat_model
    except Exception:
        # fallback: try dino_model
        from dino_model import load_dino as load_dinov2_feature_extractor
        feat_model, meta = load_dinov2_feature_extractor(extractor_path or '', device)
        return feat_model


def show_examples_for_class(data_root, cls, slot):
    # show photo and herbarium side-by-side when available
    photo_dir = Path(data_root) / 'train' / 'photo' / str(cls)
    herb_dir = Path(data_root) / 'train' / 'herbarium' / str(cls)
    cols = slot.columns(2)
    shown = False
    if photo_dir.exists() and any(photo_dir.iterdir()):
        f = next(photo_dir.iterdir())
        try:
            cols[0].image(Image.open(f).convert('RGB'), use_column_width=True)
            cols[0].caption('Photo')
            shown = True
        except Exception:
            cols[0].write('Photo load failed')
    else:
        cols[0].write('No photo')

    if herb_dir.exists() and any(herb_dir.iterdir()):
        f2 = next(herb_dir.iterdir())
        try:
            cols[1].image(Image.open(f2).convert('RGB'), use_column_width=True)
            cols[1].caption('Herbarium')
            shown = True
        except Exception:
            cols[1].write('Herbarium load failed')
    else:
        cols[1].write('No herbarium')

    if not shown:
        slot.write('No example images found for class: ' + str(cls))


if submit:
    if uploaded is None:
        st.error('Please upload an image')
    else:
        try:
            img_pil = Image.open(uploaded).convert('RGB')
            # run inference with a spinner and show detailed traceback on failure
            try:
                with st.spinner('Running inference...'):
                    if approach == 'Approach 3':
                        ckpt_arg = ckpt_path_input if 'ckpt_path_input' in locals() else ''
                        ckpt_upload_arg = ckpt_upload if 'ckpt_upload' in locals() else None
                        res = run_model_inference('Approach 3', img_pil, top_k, device=device, ckpt=ckpt_arg, ckpt_upload=ckpt_upload_arg, data_root=data_root)
                    elif approach == 'Approach 2':
                        extractor_arg = extractor_path_input if 'extractor_path_input' in locals() else ''
                        extractor_upload_arg = extractor_upload if 'extractor_upload' in locals() else None
                        classifier_arg = classifier_path_input if 'classifier_path_input' in locals() else ''
                        classifier_upload_arg = classifier_upload if 'classifier_upload' in locals() else None
                        res = run_model_inference('Approach 2', img_pil, top_k, device=device, extractor=extractor_arg, extractor_upload=extractor_upload_arg, classifier=classifier_arg, classifier_upload=classifier_upload_arg)
                    else:
                        ckpt_arg = ckpt_path_input if 'ckpt_path_input' in locals() else ''
                        ckpt_upload_arg = ckpt_upload if 'ckpt_upload' in locals() else None
                        res = run_model_inference('Approach 1', img_pil, top_k, device=device, ckpt=ckpt_arg, ckpt_upload=ckpt_upload_arg, data_root=data_root)
            except Exception as e:
                st.error('Inference failed — see details below:')
                st.text(traceback.format_exc())
                res = []

            # display results
            with pred_box:
                st.markdown('### Top predictions')
                for i, (c, s) in enumerate(res, start=1):
                    st.markdown(f'**Top-{i}**: {c} — {s*100:.2f}%')

            # show sample gallery for Top-K
            try:
                cols = sample_container.columns(min(top_k, 6))
                for j, (c, s) in enumerate(res[:min(top_k, 6)]):
                    try:
                        show_examples_for_class(data_root, c, cols[j])
                        cols[j].caption(f'{c}\n{s*100:.1f}%')
                    except Exception:
                        cols[j].write('No example')
            except Exception:
                # fallback: show single top-1 in the original slot
                try:
                    top1 = res[0][0]
                    show_examples_for_class(data_root, top1, sample_slot)
                except Exception:
                    pass

        except Exception as e:
            st.error(f'Inference failed: {e}')

st.markdown('---')
st.markdown('Hints: Approach 2 expects a sklearn `.pkl` classifier. Approach 3 and 1 require `.pt` checkpoints produced by their training code.')
