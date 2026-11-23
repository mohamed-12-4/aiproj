"""Calibrate on site6, select on the other validation sites, then freeze for test."""

import gc
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from fusion_eval.core import ConfusionCounter, health_lookup, summarize
from fusion_eval.runner import (SCORES, atomic_json, figure, make_dataset, save_bundle,
                               sha256, write_csv)
from .analysis import (cover_metrics, health_counts, joint_errors, pixel_histogram,
                       reliability, retained_pixel_risk)
from .core import (CALIBRATION_SITE, Gate, PATCH, WEIGHTS, fit_temperature, gate_features,
                   gate_grid, health_probabilities, mixture_prediction, route,
                   temperature, uncertainty_features)
from .runtime import build_models, check_native_protocol, outputs


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [*Path(__file__).parent.glob('*.py'), root / 'fusion_eval/core.py',
             root / 'fusion_eval/runner.py', root / 'post_training_compare/runtime.py',
             root / 'reinpp_training/model.py', root / 'reinpp_training/config.py',
             root / 'reinpp_training/data.py', root / 'paper_suite/recipe.py']
    return {str(p.relative_to(root)): sha256(p) for p in paths}


def records():
    return ([dict(family='raw', weight=w) for w in WEIGHTS]
            + [dict(family='calibrated', weight=w) for w in WEIGHTS]
            + [dict(family='selective', **g.record()) for g in gate_grid()])


def site_of(dataset, index):
    return dataset.pairs[index][0].parent.name


@torch.inference_mode()
def calibrate(models, dataset, status, deadline):
    indices = [i for i in range(len(dataset)) if site_of(dataset, i) == CALIBRATION_SITE]
    if len(indices) != 32:
        raise ValueError('Expected the designated 32-image calibration site')
    logp, targets, protocol = [[], []], [], None
    started = time.monotonic()
    for n, index in enumerate(indices):
        if time.monotonic() > deadline:
            raise TimeoutError('Calibration runtime cap exceeded')
        image, label = dataset[index]
        first, second, _, second_pred = outputs(models, image)
        if protocol is None:
            protocol = check_native_protocol(models, image, second_pred)
        target = torch.as_tensor(np.asarray(label, dtype=np.int64)[16::32, 16::32].copy()).flatten()
        valid = target != 0
        targets.append(target[valid])
        for bucket, probabilities in zip(logp, (first, second)):
            sample = probabilities[:, 16::32, 16::32].permute(1, 2, 0).reshape(-1, 96).cpu()
            bucket.append(sample[valid].clamp_min(1e-30).log())
        del first, second
        if (n + 1) % 8 == 0 or n + 1 == len(indices):
            status(dict(phase='calibration', images_completed=n + 1, images_total=len(indices),
                        seconds=round(time.monotonic() - started, 1)))
    targets = torch.cat(targets)
    fitted = {name: fit_temperature(torch.cat(bucket), targets)
              for name, bucket in zip(('lora', 'reinpp'), logp)}
    fitted.update(site=CALIBRATION_SITE, images=len(indices), protocol_check=protocol,
                  sampling='fixed 32px grid with offset16; native target0 excluded')
    print('Calibration complete: ' + ', '.join(f'{name} T={fitted[name]["temperature"]:g}'
                                               for name in ('lora', 'reinpp')), flush=True)
    return fitted


def prepare(first, second, first_pred, second_pred, calibration, groups):
    fc = temperature(first, calibration['lora']['temperature'])
    sc = temperature(second, calibration['reinpp']['temperature'])
    fh, sh = health_probabilities(fc, groups), health_probabilities(sc, groups)
    features = gate_features(fh, sh, first_pred, second_pred, groups)
    uncertainty, js = uncertainty_features(fh, sh, first_pred, second_pred, groups)
    return dict(first=first, second=second, first_cal=fc, second_cal=sc,
                first_pred=first_pred, second_pred=second_pred, features=features,
                uncertainty=uncertainty, health_js=js)


def prediction(maps, recipe):
    if recipe['family'] == 'selective':
        return route(maps['first_pred'], maps['second_pred'], maps['features'],
                     Gate(recipe['min_advantage'], recipe['min_entropy']))
    calibrated = recipe['family'] == 'calibrated'
    p = mixture_prediction(maps['first_cal' if calibrated else 'first'],
                           maps['second_cal' if calibrated else 'second'], recipe['weight'],
                           maps['first_pred'], maps['second_pred'])
    return p, None


@torch.inference_mode()
def evaluate(models, dataset, split, recipes, calibration, groups, status, deadline):
    count = len(recipes) + 2
    totals = torch.zeros(count, 96, 96, dtype=torch.long, device='cuda')
    by_site, previews, per_image = {}, {}, []
    pixel_risk = torch.zeros(count, 512, 2, dtype=torch.long)
    joint = {name: torch.zeros(size, dtype=torch.long) for name, size in
             (('native', 4), ('health', 4), ('disagreement_strata', 2))}
    changed = torch.zeros(count, dtype=torch.long)
    examples = set(np.linspace(0, len(dataset) - 1, 3, dtype=int).tolist()) if split == 'val' else set()
    started = previous_log = time.monotonic()
    for index, (image, label) in enumerate(dataset):
        if time.monotonic() >= deadline:
            raise TimeoutError('Bounded fusion runtime expired')
        first, second, fp, sp = outputs(models, image)
        maps = prepare(first, second, fp, sp, calibration, groups)
        target = torch.as_tensor(np.asarray(label, dtype=np.int64).copy(), device='cuda')
        counter = ConfusionCounter(target)
        site = site_of(dataset, index)
        if site not in by_site:
            by_site[site] = torch.zeros_like(totals)
        image_counts = torch.zeros_like(totals)
        modifications = torch.zeros(count, dtype=torch.long)
        this_joint = joint_errors(target, fp, sp, groups)
        for key, value in this_joint.items():
            joint[key] += value
        if index in examples:
            previews[index] = dict(image=image.resize((320, 160)),
                target=np.asarray(label.resize((320, 160), resample=0)).copy(),
                predictions=np.zeros((count, 160, 320), dtype=np.uint8))

        def add(j, pred, modified=None):
            matrix = counter(pred)
            totals[j] += matrix
            by_site[site][j] += matrix
            image_counts[j] = matrix
            pixel_risk[j] += pixel_histogram(maps['health_js'], target, pred, groups)
            if modified is not None:
                modifications[j] = int(modified.sum())
            if index in examples:
                previews[index]['predictions'][j] = F.interpolate(pred[None, None].float(),
                    size=(160, 320), mode='nearest')[0, 0].byte().cpu().numpy()

        add(0, fp)
        add(1, sp)
        for j, recipe in enumerate(recipes):
            pred, selected = prediction(maps, recipe)
            add(j + 2, pred, selected)
        cpu = image_counts.cpu()
        per_image.append(dict(site=site, image=dataset.pairs[index][0].name,
            native_stats=torch.stack((cpu.diagonal(dim1=1, dim2=2), cpu.sum(1), cpu.sum(2)), dim=1),
            health_counts=health_counts(cpu, groups), modifications=modifications,
            uncertainty=maps['uncertainty'], joint=this_joint))
        changed += modifications
        del maps, first, second, fp, sp, target, image_counts, cpu, pred, selected
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            status(dict(phase=split, images_completed=index + 1, images_total=len(dataset),
                        seconds=round(time.monotonic() - started, 1)))
        if time.monotonic() - previous_log > 120:
            print(f'Evaluation {split}: {index+1}/{len(dataset)}', flush=True)
            previous_log = time.monotonic()
    return dict(counts=totals.cpu(), by_site={s: c.cpu() for s, c in by_site.items()},
                previews=previews, per_image=per_image, joint=joint,
                pixel_risk=pixel_risk, changed=changed, seconds=time.monotonic() - started)


def image_rows(result, index, selection_only=False):
    return [dict(site=r['site'], image=r['image'], uncertainty=r['uncertainty'],
                 modified_pixels=int(r['modifications'][index]),
                 **cover_metrics(r['health_counts'][index])) for r in result['per_image']
            if not selection_only or r['site'] != CALIBRATION_SITE]


def result_part(result, index, groups):
    rows = image_rows(result, index)
    return dict(metrics=summarize(result['counts'][index], groups), counts=result['counts'][index],
                sites={s: summarize(c[index], groups) for s, c in result['by_site'].items()},
                per_image_native_stats=torch.stack([r['native_stats'][index] for r in result['per_image']]),
                per_image=rows, reliability=reliability(rows),
                health_js_pixel_histogram=result['pixel_risk'][index],
                retained80_health_pixel_risk=retained_pixel_risk(result['pixel_risk'][index]),
                modified_pixels=int(result['changed'][index]))


def choose(validation, grid, groups):
    selected_sites = {s: c for s, c in validation['by_site'].items() if s != CALIBRATION_SITE}
    if set(selected_sites) != {'site29', 'site30'}:
        raise ValueError('Unexpected fusion-selection sites')
    counts = sum(selected_sites.values())
    all_metrics = [summarize(c, groups) for c in counts]
    result = []
    for family in ('raw', 'calibrated', 'selective'):
        pool = [i for i, r in enumerate(grid) if r['family'] == family]
        def key(j):
            recipe = grid[j]
            # Exact metric ties keep LoRA or the no-routing control.
            preference = recipe.get('weight', 1. if recipe.get('min_advantage') is None else 0.)
            return all_metrics[j + 2]['native_miou'], preference
        result.append(max(pool, key=key))
    return result, all_metrics, counts


def row(name, recipe, calibration, validation_part, selection, suite_seconds):
    m = validation_part['metrics']
    return dict(epoch=1, evaluation_only=True, phase='validation_selected', model=name,
        family=recipe['family'] if recipe else 'individual',
        lora_weight=recipe.get('weight', '') if recipe else '',
        min_advantage=recipe.get('min_advantage', '') if recipe else '',
        min_entropy=recipe.get('min_entropy', '') if recipe else '',
        lora_temperature=calibration['lora']['temperature'],
        reinpp_temperature=calibration['reinpp']['temperature'],
        calibration_images=32, selection_images=138, validation_images=170, test_images=0,
        **{f'val_{k}': m[k] for k in SCORES}, selection_native_miou=selection['native_miou'],
        **{f'test_{k}': '' for k in SCORES}, test_dead_cover_mae_pp='',
        test_js_retained80_dead_cover_mae_pp='', test_js_cover_error_auroc='',
        test_js_retained80_health_pixel_risk='', suite_seconds=suite_seconds,
        peak_vram_gib='')


def run_suite(workspace, rein_checkpoint, suite, commit=lambda: None, max_seconds=5300):
    from paper_suite.recipe import DATA_REVISION
    workspace = Path(workspace)
    started = time.monotonic()
    deadline = started + max_seconds
    names = [f'{suite}-{suffix}' for suffix in ('lora', 'reinpp', 'e1-raw', 'e1-calibrated', 'e2-selective')]
    if any((workspace / 'runs' / name / 'metrics.csv').exists() for name in names):
        raise FileExistsError('Refusing to overwrite an existing fusion suite')
    frozen = {}
    def status(state):
        atomic_json(workspace / 'checkpoints/local' / f'{suite}-status.json',
                    dict(suite=suite, selected=frozen, **state))
        commit()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    status(dict(phase='preparing_models', images_completed=0))
    print('Preparing frozen epoch-40 LoRA EMA and Rein++ EMA; no TTA or model updates', flush=True)
    models, sources = build_models(workspace, rein_checkpoint)
    bn_before = {name: {k: v.detach().cpu().clone() for k, v in model.named_buffers()
        if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k}
        for name, model in models.items()}
    dataset = make_dataset(workspace, 'val')
    calibration = calibrate(models, dataset, status, deadline)
    groups_cpu = health_lookup()
    groups = groups_cpu.cuda()
    grid = records()
    validation = evaluate(models, dataset, 'val', grid, calibration, groups, status, deadline)
    chosen, selection_metrics, selection_counts = choose(validation, grid, groups_cpu)
    winners = [grid[i] for i in chosen]
    frozen.update({r['family']: r for r in winners})
    indices = [0, 1] + [i + 2 for i in chosen]
    provenance = dict(sources=sources, dataset_revision=DATA_REVISION, source_sha256=source_hashes(),
        calibration_site=CALIBRATION_SITE, selection_sites=['site29', 'site30'],
        selection_objective='highest pooled native mIoU on selection sites; exact ties keep LoRA',
        known_limit='Official validation previously used for checkpoint selection; test previously inspected, so this is exploratory follow-up',
        health_policy='native argmax then unchanged V2 health mapping; unknown predictions remain errors',
        cover_policy='Dead pixel fraction on the same health-evaluable annotated image area; absolute error in percentage points',
        uncertainty_policy='All five predictors prespecified; ROI is union of predicted coral, no GT inputs; spatial JS is 90th percentile of 32px ROI-patch means',
        routing_policy='Only health-disagreeing pixels, Rein health known, >=75% Rein health agreement within32px patch; threshold calibrated support advantage and LoRA entropy',
        trainable_model_updates=0, augmentation=False, torch=str(torch.__version__))
    bundles, rows = [], []
    for i, name in enumerate(names):
        index = indices[i]
        recipe = None if i < 2 else winners[i - 2]
        part = result_part(validation, index, groups_cpu)
        selection_rows = image_rows(validation, index, selection_only=True)
        selection = dict(metrics=selection_metrics[index], counts=selection_counts[index],
                         reliability=reliability(selection_rows), images=138)
        bundle = dict(format_version=1, checkpoint_kind='frozen_spatial_fusion_recipe', name=name,
            individual_model=('lora', 'reinpp')[i] if i < 2 else None,
            recipe=recipe, calibration=calibration, provenance=provenance,
            validation=part, selection=selection, validation_joint_errors=validation['joint'])
        if recipe:
            bundle['selection']['search'] = [dict(recipe=r,
                native_miou=selection_metrics[j + 2]['native_miou'],
                coral_miou=selection_metrics[j + 2]['coral_miou'],
                dead_iou=selection_metrics[j + 2]['dead_iou'])
                for j, r in enumerate(grid) if r['family'] == recipe['family']]
        rows.append(row(name, recipe, calibration, part, selection['metrics'], time.monotonic() - started))
        directory = workspace / 'runs' / name
        write_csv(directory, rows[-1])
        figure(directory, validation['previews'], index, groups_cpu,
               f'{name} | fixed validation examples; weights selected on sites29/30')
        save_bundle(workspace / 'checkpoints/local' / name / 'head.pt', bundle)
        bundles.append(bundle)
        m = part['metrics']
        print(f'Validation {name}: native={m["native_miou"]:.4f}; selection={selection["metrics"]["native_miou"]:.4f}'
              + (f'; recipe={recipe}' if recipe else ''), flush=True)
    commit()
    del validation
    gc.collect()
    torch.cuda.empty_cache()
    test = evaluate(models, make_dataset(workspace, 'test'), 'test', winners,
                    calibration, groups, status, deadline)
    output = []
    for i, name in enumerate(names):
        part = result_part(test, i, groups_cpu)
        bundles[i]['test'] = part
        bundles[i]['test_joint_errors'] = test['joint']
        m, a = part['metrics'], part['reliability']
        js = a['predictors']['health_js']
        rows[i].update(phase='complete', test_images=473,
            **{f'test_{key}': m[key] for key in SCORES}, test_dead_cover_mae_pp=a['dead_cover_mae_pp'],
            test_js_retained80_dead_cover_mae_pp=js['retained80_dead_cover_mae_pp'],
            test_js_cover_error_auroc=js['auroc_cover_error_gt5pp'],
            test_js_retained80_health_pixel_risk=part['retained80_health_pixel_risk'],
            suite_seconds=time.monotonic() - started, peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_csv(workspace / 'runs' / name, rows[i])
        save_bundle(workspace / 'checkpoints/local' / name / 'head.pt', bundles[i])
        output.append(dict(name=name, recipe=bundles[i]['recipe'],
            test={key: m[key] for key in SCORES}, dead_cover_mae_pp=a['dead_cover_mae_pp'],
            js_retained80_dead_cover_mae_pp=js['retained80_dead_cover_mae_pp']))
        print(f'Test {name}: native={m["native_miou"]:.4f} coral={m["coral_miou"]:.4f} Dead={m["dead_iou"]:.4f}; coverMAE={a["dead_cover_mae_pp"]:.3f}pp', flush=True)
    for name, model in models.items():
        buffers = dict(model.named_buffers())
        if any(not torch.equal(value, buffers[key].cpu()) for key, value in bn_before[name].items()):
            raise RuntimeError('Frozen evaluation changed BatchNorm buffers')
    status(dict(phase='complete', images_completed=473, images_total=473,
                seconds=round(time.monotonic() - started, 1), results=output))
    print('Final paths: ' + ', '.join('runs/' + name for name in names), flush=True)
    return dict(suite=suite, results=output, seconds=time.monotonic() - started)
