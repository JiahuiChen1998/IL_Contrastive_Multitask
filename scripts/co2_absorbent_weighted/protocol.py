"""Prespecified CO2-absorbent priorities and immutable experiment inputs."""
import hashlib
import json
from pathlib import Path
import sys

from il_contrastive.equations import study_tasks
from il_contrastive.pretraining import digest
from scripts import run_multitask_heads_study as legacy

ROOT=legacy.ROOT
OUT=Path('runs/co2_absorbent_weighted_30m_v1')
DATA=legacy.DATA
CHECKPOINT=legacy.CHECKPOINT
PRIMARY=['melting_point','co2_solubility','viscosity']
SEEDS=list(range(42,47))
ARMS={'uniform_plain':dict(priority='uniform'),'weighted_plain':dict(priority='co2_90')}
FILES=['best.pt','metrics.json','task_config.json','validation_predictions.csv','test_predictions.csv','execution.json']


def loss_weights(priority,tasks=None):
    tasks=list(study_tasks()) if tasks is None else list(tasks)
    if set(tasks)!=set(study_tasks()) or len(tasks)!=11:
        raise ValueError('Retain all eleven unique regression tasks')
    if priority=='uniform':
        return {t:1/len(tasks) for t in tasks}
    if priority!='co2_90':
        raise ValueError('Unknown priority')
    return {t:.9/len(PRIMARY) if t in PRIMARY else .1/(len(tasks)-len(PRIMARY)) for t in tasks}


def fingerprint():
    files=[Path(__file__),Path(__file__).with_name('train.py')]
    return hashlib.sha256(legacy.training_fingerprint().encode()+b''.join(p.read_bytes() for p in files)).hexdigest()


def make_plan(output=OUT,smoke=False):
    output=Path(output)
    parent=json.loads((legacy.OUT/'plan.json').read_text())
    if parent['source_sha256']!=legacy.training_fingerprint():
        raise ValueError('Original scientific source changed')
    if digest(DATA)!=parent['data_sha256'] or digest(CHECKPOINT)!=parent['checkpoint_sha256']:
        raise ValueError('Frozen data or checkpoint changed')
    settings=dict(hidden=64,layers=3,heads=4,cross_ion=True,head_mode='mlp',
                  epochs=1 if smoke else 100,steps_per_epoch=2 if smoke else 100,
                  per_task_batch=2 if smoke else 8,lr=.0005,weight_decay=.0001,patience=15,
                  threads=4,gradient_clip=5.,device='cpu' if smoke else 'cuda',
                  limit_per_task=8 if smoke else 0,task_schema='ablation_v1')
    seeds=[42] if smoke else SEEDS
    jobs=[]
    for seed in seeds:
        for arm,design in ARMS.items():
            jobs.append(dict(id=f'{seed}/{arm}',seed=seed,arm=arm,**design,
                             output=str(output/'fits'/f'seed_{seed}'/arm),weights=loss_weights(design['priority']),
                             command=[sys.executable,'-u','-m','scripts.co2_absorbent_weighted.train',
                                      '--root',str(output),'--job',f'{seed}/{arm}']))
    refs=[]
    if not smoke:
        for j in parent['jobs']:
            if j['initialization']!='pretrained_30m':continue
            historical=j['scope']=='multi' and j['head_family']=='mlp'
            single=j['scope']=='single' and j['tasks'][0] in PRIMARY and 'mlp' in j['head_aliases']
            if not (historical or single):continue
            if not legacy.verify_fit(parent,j):raise ValueError(f'Missing reference: {j["id"]}')
            path=Path(j['output'])
            files=legacy.ARTIFACTS+(['execution.json'] if (path/'execution.json').exists() else [])
            refs.append(dict(id=j['id'],seed=j['seed'],tasks=j['tasks'],output=str(path),
                             arm='historical_multi' if historical else 'single_task',
                             hashes={name:digest(path/name) for name in files}))
    return dict(schema='co2_absorbent_weighted_30m_v1',smoke=smoke,primary=PRIMARY,
                tasks=study_tasks(),seeds=seeds,settings=settings,arms=ARMS,jobs=jobs,references=refs,
                data_sha256=digest(DATA),manifest_sha256=digest(DATA.parent/'manifest.json'),
                split_sha256=digest(DATA.parent/'structure_splits.csv'),
                checkpoint=str(CHECKPOINT),checkpoint_sha256=digest(CHECKPOINT),source_sha256=fingerprint(),
                legacy_plan_sha256=digest(legacy.OUT/'plan.json'),
                selection='One checkpoint: equal mean validation standardized MSE of MP, ln(CO2 solubility), ln(viscosity).',
                targets='All eleven continuous regressions; no application thresholds or classification targets.',
                protocol='User-chosen 30.2M general-molecular initialization; all eleven MLP tasks; original cross-ion configuration retained. '
                'Uniform versus CO2-application-weighted loss only; no added task attention or adapter. '
                'Five seeds on the frozen complete-IL split. Both new arms use the same three-property validation criterion. '
                'Historical eleven-task-selected multitask and property-selected single-task baselines remain separate. '
                'CO2/viscosity are trained in ln space; native CO2 MAE and both transformed MAEs are reported. '
                'No test selection; prior test exposure disclosed.')


def verify_inputs(plan):
    for path,key in [(DATA,'data_sha256'),(DATA.parent/'manifest.json','manifest_sha256'),
                     (DATA.parent/'structure_splits.csv','split_sha256'),(CHECKPOINT,'checkpoint_sha256'),
                     (legacy.OUT/'plan.json','legacy_plan_sha256')]:
        if digest(path)!=plan[key]:raise ValueError(f'Changed immutable input: {path}')
    if fingerprint()!=plan['source_sha256']:
        raise ValueError('Scientific source changed; choose a separately versioned plan')


def verify_fit(plan,job):
    path=Path(job['output'])
    if not (path/'execution.json').exists():return False
    execution=json.loads((path/'execution.json').read_text())
    if execution['state']!='complete':return False
    if not all((path/name).exists() for name in FILES):raise ValueError(f'Incomplete artifacts: {path}')
    m=json.loads((path/'metrics.json').read_text())
    expected=dict(settings=plan['settings'],seed=job['seed'],arm=job['arm'],weights=job['weights'])
    if m['arguments']!=expected or m['source_sha256']!=plan['source_sha256']:
        raise ValueError(f'Configuration or source mismatch: {path}')
    if m['data_sha256']!=plan['data_sha256'] or bool(m['subset_run'])!=plan['smoke']:
        raise ValueError(f'Data/subset mismatch: {path}')
    if m['pretraining']['sha256']!=plan['checkpoint_sha256'] or m['pretraining']['include_cross']:
        raise ValueError(f'Initialization mismatch: {path}')
    if execution['source_sha256']!=plan['source_sha256']:raise ValueError(f'Execution source mismatch: {path}')
    cfg=json.loads((path/'task_config.json').read_text())
    expected_config={k:plan['settings'][k] for k in ['hidden','layers','heads','cross_ion','head_mode']}
    expected_config['tasks']=plan['tasks']
    if cfg!=dict(tasks=plan['tasks'],weights=job['weights'],selection_tasks=PRIMARY,config=expected_config):
        raise ValueError(f'Task/model/selection mismatch: {path}')
    if m['trainable_parameters']!=384803:raise ValueError(f'Expected the unchanged eleven-task MLP: {path}')
    return True
