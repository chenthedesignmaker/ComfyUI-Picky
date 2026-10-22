import copy
import json
from pathlib import Path

import pytest

from video_prompter import video_loras
from video_prompter.api import Api, ApiError
from video_prompter.gallery import Take, takes_from_history
from video_prompter.profiles import ProfileStore
from video_prompter.projects import ProjectStore

STACK = [{"name": "people.safetensors", "strength_model": 0.6, "enabled": True},
         {"name": "cinema.safetensors", "strength_model": 0.2, "enabled": True}]


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(video_loras, "installed_names", lambda: [r['name'] for r in STACK])
    api = Api(projects=ProjectStore(tmp_path/'projects'), profiles=ProfileStore(tmp_path/'profiles'))
    api.create_project({"name": "Test"})
    api.add_source({"project": "Test", "name": "First shot"})
    project = api.projects.load("Test"); project.settings['video_loras'] = STACK
    api.projects.save(project)
    workflow = json.loads((Path(__file__).parent/'fixtures'/'h3_i2v_real.json').read_text())
    api.save_profile({"name": "test", "api_format": workflow})
    return api


def prompt():
    return {"duration": 5, "style": "Live-action, r34l1sm", "audio": "Quiet room tone.",
            "beats": [{"start": 0, "end": 5, "angle": "wide shot", "action": "a woman sits at her desk"}]}


def request():
    return {"project": "Test", "profile": "test", "quality": "final", "prompt": prompt(), "seed": 7}


def assert_stack(result):
    assert result['loras'] == STACK
    graph = result['workflow']
    nodes = {k:n for k,n in graph.items() if k.startswith('vp_video_lora_')}
    assert len(nodes) == 2
    first, second = nodes.values()
    root = first['inputs']['model']
    assert graph[root[0]]['class_type'] == 'UNETLoader'
    assert second['inputs']['model'] == [next(iter(nodes)), 0]
    # No conditioning/sampler/second pass may bypass the patched model.
    consumers = [k for k,n in graph.items() if root in n.get('inputs',{}).values()]
    assert consumers == [next(iter(nodes))]


@pytest.mark.parametrize('mode', ['t2v', 'i2v', 'ref2v'])
@pytest.mark.parametrize('tier', ['final', 'final_2pass', 'selflift_final'])
def test_stack_reaches_all_modes_and_passes(api, mode, tier):
    body=request(); body['quality']=tier
    if mode=='i2v': body['prompt']['first_frame']='first.png'
    if mode=='ref2v':
        member=api.save_member({'project':'Test','member':{'name':'Alice','kind':'character','references':[{'kind':'image','path':'alice.png'}]}})['member']
        body['prompt']['cast']=[member['id']]
    result=api.prepare_render(body)
    assert result['mode']==mode
    assert_stack(result)


def test_reshoot_and_batches(api):
    body={**request(),'prompt_text':'A woman at her desk.','width':1344,'height':736,'frames':124,'seconds':5.167}
    assert_stack(api.prepare_reshoot(body))
    for r in api.prepare_batch({**request(),'count':2})['renders']: assert_stack(r)
    body.update(source='video/parent.mp4 [output]')
    assert_stack(api.prepare_extend(body))
    for r in api.prepare_extend_batch({**body,'count':2})['renders']: assert_stack(r)


def test_explicit_bypass_and_disabled_or_zero_are_no_ops(api):
    plain=api.prepare_render({**request(),'loras':[]})
    muted=copy.deepcopy(STACK); muted[0]['enabled']=False; muted[1]['strength_model']=0
    result=api.prepare_render({**request(),'loras':muted})
    assert plain['loras']==result['loras']==[]
    assert not any(k.startswith('vp_video_lora_') for k in result['workflow'])
    assert api.projects.load('Test').settings['video_loras']==STACK


def test_settings_update_preserves_project_and_rejects_invalid_values(api):
    before=api.projects.load('Test').to_dict()
    result=api.update_video_loras({'project':'Test','loras':[]})
    assert result=={'loras':[]}
    after=api.projects.load('Test').to_dict()
    assert after['sources']==before['sources'] and after['timeline']==before['timeline']
    assert after['settings']['video_loras']==[]
    with pytest.raises(ApiError): api.update_video_loras({'project':'Test','loras':[{'name':'x','strength_model':99}]})
    assert api.projects.load('Test').settings['video_loras']==[]


@pytest.mark.parametrize('stack', [None, {}, [None], [{'name':'x','strength_model':float('nan')}],
    [{'name':'x','strength_model':float('inf')}], [{'name':'x','strength_model':True}],
    [{'name':'../x'}], [{'name':'C:/x'}], [STACK[0],STACK[0]], [{'name':'missing.safetensors'}],
    [{'name':'turbo.safetensors'}], [{'name':'x','strength_model':3}]])
def test_invalid_stacks_fail_before_queue(api, stack):
    with pytest.raises(ApiError) as error: api.prepare_render({**request(),'loras':stack})
    assert error.value.status==400


def test_extend_inherits_recorded_parent_stack_and_explicit_override_wins(api):
    parent=Take(prompt_id='parent',filename='parent.mp4',loras=[])
    api.gallery.record('Test',[parent])
    body={**request(),'source':'video/parent.mp4 [output]','parent_take':parent.id,
          'prompt_text':'She stands up.','width':1344,'height':736,'frames':124,'seconds':5.167}
    assert api.prepare_extend(body)['loras']==[]
    assert_stack(api.prepare_extend({**body,'loras':STACK}))


def test_recorded_stack_round_trip_and_legacy_unknown():
    entry={'prompt':[0,'job',{}, {'video_prompter':{'loras':STACK}}],
           'outputs':{'save':{'videos':[{'filename':'test.mp4'}]}}}
    take=takes_from_history('job',entry)[0]
    assert Take.from_dict(take.to_dict()).loras==STACK
    assert Take.from_dict({'filename':'legacy.mp4'}).loras is None
    assert Take.from_dict({'filename':'plain.mp4','loras':[]}).loras==[]


def test_no_mutation_and_preserves_existing_turbo_and_clip(monkeypatch):
    monkeypatch.setattr(video_loras,'installed_names',lambda:None)
    graph={'model':{'class_type':'CheckpointLoaderSimple','inputs':{}},
           'turbo':{'class_type':'LoraLoaderModelOnly','inputs':{'model':['model',0],'lora_name':'turbo','strength_model':0}},
           'clip':{'class_type':'CLIPTextEncode','inputs':{'clip':['model',1]}}}
    before=copy.deepcopy(graph);patched,active=video_loras.apply(graph,STACK)
    assert graph==before and active==STACK
    assert patched['turbo']['inputs']['strength_model']==0
    assert patched['clip']['inputs']['clip']==['model',1]
    assert patched['turbo']['inputs']['model']==['vp_video_lora_1',0]


def shot_with_loras(api, mode, rows=None):
    source = api.projects.load('Test').sources[0].to_dict()
    source['segments'][0]['prompt'] = prompt()
    source['video_loras'] = {'mode': mode, 'loras': rows if rows is not None else STACK[:1]}
    api.update_source({'project': 'Test', 'source': source})
    return source


@pytest.mark.parametrize('mode, expected', [('inherit', STACK), ('custom', STACK[:1]), ('off', [])])
def test_shot_stack_resolution_across_render_paths(api, mode, expected):
    source = shot_with_loras(api, mode)
    body = {**request(), 'source_id': source['id']}
    assert api.prepare_render(body)['loras'] == expected
    assert all(r['loras'] == expected for r in api.prepare_batch({**body, 'count': 2})['renders'])
    body.update(prompt_text='She stands up.', width=1344, height=736, frames=124, seconds=5.167)
    assert api.prepare_reshoot(body)['loras'] == expected
    body['source'] = 'video/parent.mp4 [output]'
    assert api.prepare_extend(body)['loras'] == expected
    assert all(r['loras'] == expected for r in api.prepare_extend_batch({**body, 'count': 2})['renders'])
    assert api.prepare_render({**request(), 'source_id': source['id'], 'loras': []})['loras'] == []
    assert api.prepare_render({**request(), 'source_id': source['id'], 'loras': STACK})['loras'] == STACK


@pytest.mark.parametrize('mode, expected', [('inherit', STACK[1:]), ('custom', STACK[:1]), ('off', [])])
def test_shot_override_beats_parent_but_inherit_retains_continuity(api, mode, expected):
    source = shot_with_loras(api, mode)
    parent = Take(prompt_id='parent', filename='parent.mp4', loras=STACK[1:])
    api.gallery.record('Test', [parent])
    body = {**request(), 'source_id': source['id'], 'parent_take': parent.id,
            'source': 'video/parent.mp4 [output]', 'prompt_text': 'She stands up.',
            'width': 1344, 'height': 736, 'frames': 124, 'seconds': 5.167}
    assert api.prepare_extend(body)['loras'] == expected


def test_shot_settings_survive_reload_copy_split_and_do_not_affect_other_shots(api):
    from video_prompter.sources import Source, split_off
    source = shot_with_loras(api, 'custom')
    saved = api.projects.load('Test').source(source['id'])
    assert saved.video_loras == source['video_loras']
    copied = saved.copy()
    split = split_off(saved, saved.segments[0], [0])
    assert copied.video_loras == split.video_loras == saved.video_loras
    copied.video_loras['loras'][0]['strength_model'] = 1.5
    assert saved.video_loras['loras'][0]['strength_model'] == 0.6
    other = Source()
    api.projects.update('Test', lambda p: p.add_source(other))
    assert api.prepare_render({**request(), 'source_id': other.id})['loras'] == STACK
    assert api.projects.load('Test').settings['video_loras'] == STACK
    for mode in ['off', 'inherit', 'custom']:
        source['video_loras']['mode'] = mode
        api.update_source({'project': 'Test', 'source': source})
        assert api.projects.load('Test').source(source['id']).video_loras == source['video_loras']


@pytest.mark.parametrize('setting', [[], 'off', {'mode': []}, {'mode': 'bad'},
    {'mode': 'custom', 'loras': [{'name': 'people.safetensors', 'strength_model': 3}]}])
def test_invalid_shot_settings_rejected_without_saving(api, setting):
    source = api.projects.load('Test').sources[0].to_dict()
    source['video_loras'] = setting
    with pytest.raises(ApiError) as error:
        api.update_source({'project': 'Test', 'source': source})
    assert error.value.status == 400
    assert api.projects.load('Test').source(source['id']).video_loras is None


def test_missing_source_is_not_silently_rendered_with_project_defaults(api):
    with pytest.raises(ApiError) as error:
        api.prepare_render({**request(), 'source_id': 'does-not-exist'})
    assert error.value.status == 404


def test_catalog_uses_metadata_not_filename_and_excludes_turbo(tmp_path, monkeypatch):
    import struct
    import sys
    import types
    files = {
        'renamed-character.safetensors': {'base_model': 'minimax-h3-fl2va'},
        'cinema.safetensors': {'ss_base_model_version': 'minimax_h3'},
        'motion.safetensors': {'modelspec.architecture': 'minimax_h3_fl2v'},
        'minimax_h3_wrong.safetensors': {'base_model': 'qwen-image'},
        'unknown.safetensors': {},
        'minimax_h30.safetensors': {'base_model': 'minimax-h30'},
        'turbo_8step.safetensors': {'base_model': 'minimax-h3'},
        'training_checkpoint.pt': {'base_model': 'minimax-h3'},
    }
    for name, metadata in files.items():
        header = json.dumps({'__metadata__': metadata}).encode()
        (tmp_path/name).write_bytes(struct.pack('<Q', len(header)) + header)
    (tmp_path/'broken.safetensors').write_bytes(b'bad')
    (tmp_path/'oversized.safetensors').write_bytes(struct.pack('<Q', 100_000_000))
    names = list(files) + ['broken.safetensors', 'oversized.safetensors', 'missing.safetensors']
    monkeypatch.setattr(video_loras, 'installed_names', lambda: names)
    monkeypatch.setitem(sys.modules, 'folder_paths', types.SimpleNamespace(
        get_full_path=lambda kind, name: str(tmp_path/name)))
    result = video_loras.h3_catalog()
    assert result['loras'] == names[:3]
    assert result['hidden_count'] == len(names) - 3
    assert result['model_family'] == 'minimax-h3'
    # A changed file is re-inspected after Refresh, rather than cached forever.
    (tmp_path/names[0]).write_bytes(b'changed')
    assert video_loras.h3_catalog()['loras'] == names[1:3]


def test_catalog_outside_comfyui(monkeypatch):
    monkeypatch.setattr(video_loras, 'installed_names', lambda: None)
    assert video_loras.h3_catalog() == {
        'loras': [], 'available': False, 'model_family': 'minimax-h3', 'hidden_count': 0}


def test_bypassed_profile_lora_and_tuple_wires(monkeypatch):
    monkeypatch.setattr(video_loras, 'installed_names', lambda: None)
    graph = {
        'model': {'class_type': 'UNETLoader', 'inputs': {}},
        'bypassed': {
            'class_type': 'LoraLoaderModelOnly',
            'inputs': {'model': ('model', 0), 'lora_name': 'people.safetensors', 'strength_model': 0.0},
        },
        'guider': {'class_type': 'BasicGuider', 'inputs': {'model': ('model', 0)}},
    }
    patched, active = video_loras.apply(graph, STACK[:1])
    assert len(active) == 1
    assert patched['guider']['inputs']['model'] == ['vp_video_lora_0', 0]

