"""Entry point for the naive computer-generated holography task."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.actor.pg import PG
from env.holography_env_sim import HoloEnvSim
from tasks.param.param_holo import settings, holo_param, actor_param, optics_param
from trainer.mfo_holography import MFO
from utils.cgh_utils import RewardFn, load_target
from utils.tensor_utils import normalize
from torchvision.utils import save_image
import os


def main_cgh(settings, holo_param, actor_param, optics_param):
    if settings['env_name'] != 'sim':
        raise NotImplementedError("Only simulated CGH was moved into this repo.")
    if settings['optimizer'] != 'mfo' or settings['pg_type'] != 'loo':
        raise NotImplementedError("Only the naive model-free 'loo' CGH method is included.")

    in_size = holo_param['in_size']
    partition = holo_param['partition']

    target = load_target(
        dataset_name=settings['dataset_name'],
        in_size=in_size,
        sample_idx=holo_param['sample_idx'],
    )
    reward_fn = RewardFn(metric='mse', target=normalize(target))

    if holo_param['save_target']:
        os.makedirs('data', exist_ok=True)
        save_image(normalize(target, minmax=True), 'data/cgh_target.jpg')

    env = HoloEnvSim(
        input_dx=optics_param['input_dx'],
        input_field_shape=[in_size, in_size],
        output_dx=optics_param['output_dx'],
        output_field_shape=[in_size, in_size],
        wave_lengths=optics_param['wave_lengths'],
        z=optics_param['z'],
        response_type=optics_param['response_type'],
        pad_scale=optics_param['pad_scale'],
        sensor_effective_shape=[in_size, in_size],
        slm_size=in_size,
        partition=partition,
        slm_type=settings['slm_type'],
    )

    actor = PG(
        mask_shape=[partition, partition],
        query_batchsize=actor_param['maskquery_batchsize'],
        pg_lr=actor_param['pg_lr'],
        dp_std=actor_param['dp_std'],
        use_scheduler=actor_param['use_scheduler'],
        optimizer_type=actor_param['optimizer_type'],
        output_normalize_flag=actor_param['output_normalize_flag'],
    )

    mfo = MFO(
        macro_itr=holo_param['macro_itr'],
        slm_type=settings['slm_type'],
        env=env,
        actor=actor,
        reward_fn=reward_fn,
        query_batchsize=actor_param['maskquery_batchsize'],
        pg_type=settings['pg_type'],
        output_normalize_flag=actor_param['output_normalize_flag'],
        show_result=holo_param['show_result'],
        show_interval=holo_param['show_interval'],
    )
    return mfo.optimize()


if __name__ == "__main__":
    main_cgh(settings, holo_param, actor_param, optics_param)
