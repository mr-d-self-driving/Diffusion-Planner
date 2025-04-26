from typing import Dict

import torch

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm


@torch.no_grad()
def dpm_sampler(
    model: torch.nn.Module,
    x_T: torch.Tensor,
    other_model_params: Dict,
    model_wrapper_params: Dict,
    dpm_solver_params: Dict,
):
    noise_schedule = dpm.NoiseScheduleVP(schedule="linear")

    model_fn = dpm.model_wrapper(
        model,  # use your noise prediction model here
        noise_schedule,
        model_type=model.model_type,  # or "x_start" or "v" or "score"
        model_kwargs=other_model_params,
        **model_wrapper_params,
    )

    dpm_solver = dpm.DPM_Solver(
        model_fn, noise_schedule, algorithm_type="dpmsolver++", **dpm_solver_params
    )  # w.o. dynamic thresholding

    # Steps in [10, 20] can generate quite good samples.
    sample_dpm = dpm_solver.sample(
        x_T,
        steps=10,
        skip_type="logSNR",
        method="multistep",
        denoise_to_zero=True,
    )

    return sample_dpm
