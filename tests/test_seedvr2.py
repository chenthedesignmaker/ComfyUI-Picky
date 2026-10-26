"""SeedVR2 delivery, isolation and face-mask preservation regressions."""
import pytest
from video_prompter.refine import RefineSpec, prepare_refine
from video_prompter.seedvr2 import output_canvas, face_mask, smoothed_boxes, SeedVR2FaceBlend


@pytest.mark.parametrize('mode,res,size,tags', [
    ('seedvr2','original',(1344,736),['seedvr2']),
    ('seedvr2','1080p',(1972,1080),['seedvr2','upscale']),
    ('seedvr2_faces','original',(1344,736),['seedvr2','face_blend']),
])
def test_modes_keep_audio_timing_and_never_load_h3(mode,res,size,tags):
    spec=RefineSpec.from_request({'enhancement_mode':mode,'enhancement_resolution':res})
    p=prepare_refine(source_video='take.mp4',source_take_id='original',source_width=1344,
                     source_height=736,spec=spec)
    assert (p.width,p.height)==size
    assert p.refinement==tags and p.source_take_id=='original'
    wf=p.workflow
    assert not any('H3' in n['class_type'] for n in wf.values())
    assert wf['8']['inputs']['audio']==['2',1]
    assert wf['8']['inputs']['frame_rate']==['2',2]
    assert wf['8']['inputs']['trim_to_audio'] is False
    assert wf['6']['inputs']['width']==p.width and wf['6']['inputs']['height']==p.height
    temporal=wf['5']['inputs']
    assert (temporal['batch_size'],temporal['temporal_overlap'],temporal['prepend_frames'])==(21,5,4)
    assert temporal['uniform_batch_size'] is True
    assert temporal['input_noise_scale']==temporal['latent_noise_scale']==0
    if mode=='seedvr2_faces':
        assert wf['7']['inputs']['original']==['2',0]
        assert wf['7']['inputs']['enhanced']==['6',0]
        assert wf['8']['inputs']['images']==['7',0]
    else: assert '7' not in wf


def test_a_64_bit_h3_seed_fits_seedvr2s_32_bit_range():
    spec=RefineSpec.from_request({'enhancement_mode':'seedvr2','enhancement_resolution':'1080p'})
    p=prepare_refine(source_video='take.mp4',source_take_id='original',source_width=576,
                     source_height=352,spec=spec,seed=7641838974680053)
    assert 0 <= p.workflow['5']['inputs']['seed'] <= 2**32 - 1


def test_original_size_is_exact_and_upscale_never_downsizes():
    assert output_canvas(1280,720,'original')==(1280,720)
    assert output_canvas(3840,2160,'1080p')==(3840,2160)
    assert output_canvas(720,1280,'1080p')==(1080,1920)
    with pytest.raises(ValueError): output_canvas(0,736,'original')
    with pytest.raises(ValueError): output_canvas(1344,736,'4k_typo')


@pytest.mark.parametrize('extra',[
    {'enhancement_mode':'typo'},
    {'enhancement_mode':'seedvr2_faces','enhancement_resolution':'1080p'},
    {'enhancement_mode':'seedvr2','reference_face':'someone.png'},
])
def test_unsupported_options_fail_instead_of_silently_changing_behavior(extra):
    with pytest.raises(ValueError):
        prepare_refine(source_video='take.mp4',source_take_id='a',source_width=1344,
                       source_height=736,spec=RefineSpec.from_request(extra))


def test_mask_is_local_feathered_and_overlap_does_not_amplify():
    np=pytest.importorskip('numpy');pytest.importorskip('cv2')
    box=[30,20,50,50]
    m=face_mask([box],100,80)
    assert m[35,40]==1 and np.count_nonzero((m>0)&(m<1))>0
    assert np.all(m[:10]==0) and np.all(m[:,60:]==0)
    assert np.array_equal(m,face_mask([box,box],100,80))
    rng=np.random.default_rng(0);original=rng.random((80,100,3));enhanced=rng.random(original.shape)
    blended=original*(1-m[:,:,None])+enhanced*m[:,:,None]
    assert np.array_equal(blended[m==0],original[m==0])
    assert np.array_equal(blended[m==1],enhanced[m==1])
    assert np.all(face_mask([[-100,-100,-90,-90]],100,80)==0)


def test_smoothing_does_not_fill_dropouts_or_cross_cuts():
    pytest.importorskip('scipy')
    b=[30,20,50,50]
    detections=[[b]]*3+[[]]+[[b]]*3+[[[35,20,55,50]]]*3
    frames,n,total=smoothed_boxes(detections,cuts=[7])
    assert frames[3]==[]
    assert n==total==2
    assert frames[6][0][0]==30 and frames[7][0][0]==35


def test_node_rejects_mismatched_frames_before_detecting(monkeypatch):
    # Minimal runtime modules keep this guard independently testable.
    import sys
    from types import SimpleNamespace
    torch=pytest.importorskip('torch')
    monkeypatch.setitem(sys.modules,'folder_paths',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'comfy',SimpleNamespace(model_management=SimpleNamespace()))
    monkeypatch.setitem(sys.modules,'comfy.model_management',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=None))
    with pytest.raises(ValueError,match='matching RGB'):
        SeedVR2FaceBlend().blend(torch.zeros(3,20,20,3),torch.zeros(4,20,20,3))
