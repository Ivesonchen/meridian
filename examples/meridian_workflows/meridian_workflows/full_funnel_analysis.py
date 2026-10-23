# Copyright 2026 The Meridian Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full-funnel analysis from demo/Meridian_Full_Funnel.ipynb, cells 21/29/30/32.

These standalone definitions retain the notebook's experimental two-stage
equations. Importing this module does not read or execute a notebook.
"""

import dataclasses
from typing import Any, Mapping, Optional, Sequence

import altair as alt
import arviz as az
from meridian import backend
from meridian import constants
from meridian.analysis import analyzer
from meridian.analysis import tensors
from meridian.model import equations
from meridian.model import model
from meridian.templates import formatter
import numpy as np
import pandas as pd


def _map_new_data_to_mediator(
    new_data: tensors.DataTensors | None,
    model_context_2s: Any,
    model_context_1s: Any,
) -> tensors.DataTensors | None:
  """Map new_data variables from stage 2 so they can be passed to stage 1."""
  if new_data is None:
    return None

  mapped_kwargs = {}

  def map_tensor(tensor_name, channels_2s, channels_1s, historical_tensor_1s):
    tensor_2s = getattr(new_data, tensor_name)
    if tensor_2s is None:
      return None

    tensors_list = []
    is_backend_tensor = isinstance(tensor_2s, backend.Tensor)

    for ch in channels_1s:
      if ch in channels_2s:
        idx = list(channels_2s).index(ch)
        tensors_list.append(tensor_2s[..., idx : idx + 1])
      else:
        idx = list(channels_1s).index(ch)
        historical_slice = historical_tensor_1s[..., idx : idx + 1]
        if is_backend_tensor:
          historical_slice = backend.to_tensor(
              historical_slice, dtype=backend.float_dtype
          )

        # Broadcast historical_slice shape to match tensor_2s batch dims if
        # optimizer uses batch evaluation
        if (
            hasattr(tensor_2s, "shape")
            and tensor_2s.shape[:-1] != historical_slice.shape[:-1]
        ):
          if is_backend_tensor:
            historical_slice = backend.broadcast_to(
                historical_slice, tensor_2s.shape[:-1] + (1,)
            )
          else:
            historical_slice = np.broadcast_to(
                historical_slice, tensor_2s.shape[:-1] + (1,)
            )

        tensors_list.append(historical_slice)

    if is_backend_tensor:
      return backend.concatenate(tensors_list, axis=-1)
    else:
      return np.concatenate(tensors_list, axis=-1)

  # Defensively extract channel names only when they exist
  def get_channels(channel_coord):
    return channel_coord.values if channel_coord is not None else []

  mapped_kwargs["media"] = map_tensor(
      "media",
      get_channels(model_context_2s.input_data.media_channel),
      get_channels(model_context_1s.input_data.media_channel),
      model_context_1s.input_data.media,
  )
  mapped_kwargs["media_spend"] = map_tensor(
      "media_spend",
      get_channels(model_context_2s.input_data.media_channel),
      get_channels(model_context_1s.input_data.media_channel),
      model_context_1s.input_data.media_spend,
  )
  mapped_kwargs["reach"] = map_tensor(
      "reach",
      get_channels(model_context_2s.input_data.rf_channel),
      get_channels(model_context_1s.input_data.rf_channel),
      model_context_1s.input_data.reach,
  )
  mapped_kwargs["frequency"] = map_tensor(
      "frequency",
      get_channels(model_context_2s.input_data.rf_channel),
      get_channels(model_context_1s.input_data.rf_channel),
      model_context_1s.input_data.frequency,
  )
  mapped_kwargs["rf_spend"] = map_tensor(
      "rf_spend",
      get_channels(model_context_2s.input_data.rf_channel),
      get_channels(model_context_1s.input_data.rf_channel),
      model_context_1s.input_data.rf_spend,
  )
  mapped_kwargs["time"] = new_data.time

  return tensors.DataTensors(**mapped_kwargs)


class AnalyzerFullFunnel(analyzer.Analyzer):
  """Analyzer for full-funnel Meridian models.

  This class extends the `Analyzer` class for full-funnel inference, allowing
  the computation of total incremental outcome, which includes both
  indirect effects through the mediator variables and remaining
  effects from the primary Stage 2 model.
  """

  def __init__(
      self,
      meridian: model.Meridian,
      mediator_models: Mapping[str, model.Meridian],
      inference_data: Optional[az.InferenceData] = None,
      inference_data_mediators: Optional[Mapping[str, az.InferenceData]] = None,
  ):
    super().__init__(meridian=meridian, inference_data=inference_data)
    self._mediator_models = mediator_models
    self._mediators = list(mediator_models.keys())

    if inference_data_mediators is None:
      inference_data_mediators = {}

    self._analyzer_mediators = {
        name: analyzer.Analyzer(
            meridian=mediator_model,
            inference_data=inference_data_mediators.get(name),
        )
        for name, mediator_model in mediator_models.items()
    }
    self._validate_two_stage_requirements()

  def _validate_two_stage_requirements(self):
    """Validates the specific requirements for AnalyzerFullFunnel."""
    input_data_2s = self.model_context.input_data
    saturation_spec = self.model_context.model_spec.saturation_spec
    inf_data_2s = self.inference_data

    media_channels_2s = set(input_data_2s.get_all_paid_channels())

    for mediator in self._mediators:
      mediator_model = self._mediator_models[mediator]
      analyzer_mediator = self._analyzer_mediators[mediator]

      if (
          input_data_2s.organic_media_channel is None
          or mediator not in input_data_2s.organic_media_channel.values
      ):
        raise ValueError(
            f"Mediator '{mediator}' must be included as an `organic_media` "
            "variable in the Stage 2 `meridian` model."
        )

      if isinstance(saturation_spec, str):
        if saturation_spec != "none":
          raise ValueError(
              f"Saturation for mediator '{mediator}' in Stage 2 must be 'none'."
          )
      elif saturation_spec.get(mediator) != "none":
        raise ValueError(
            f"Saturation for mediator '{mediator}' in Stage 2 must be 'none'."
        )

      if not np.array_equal(
          input_data_2s.geo.values,
          mediator_model.model_context.input_data.geo.values,
      ):
        raise ValueError(
            f"Geos mismatch. Stage 2 has {len(input_data_2s.geo.values)} geos, "
            f"but Stage 1 ('{mediator}') has "
            f"{len(mediator_model.model_context.input_data.geo.values)}."
        )

      if not np.array_equal(
          input_data_2s.time.values,
          mediator_model.model_context.input_data.time.values,
      ):
        raise ValueError(
            f"Timepoints mismatch between Stage 2 and Stage 1 ('{mediator}')."
        )

      media_channels_1s = set(
          mediator_model.model_context.input_data.get_all_paid_channels()
      )
      if not media_channels_1s.issubset(media_channels_2s):
        raise ValueError(
            "All paid media channels in the mediator models must be present in"
            " the second stage model `meridian`."
        )

      inf_data_1s = analyzer_mediator.inference_data
      for attr in ["posterior", "prior"]:
        if hasattr(inf_data_2s, attr) and hasattr(inf_data_1s, attr):
          group_2s = getattr(inf_data_2s, attr)
          group_1s = getattr(inf_data_1s, attr)
          if (
              group_2s.chain.size != group_1s.chain.size
              or group_2s.draw.size != group_1s.draw.size
          ):
            raise ValueError(
                f"MCMC traces mismatch. Stage 2 {attr} has "
                f"({group_2s.chain.size} chains, {group_2s.draw.size} draws). "
                f"Stage 1 {attr} has "
                f"({group_1s.chain.size} chains, {group_1s.draw.size} draws)."
            )

  def _get_mediator_metadata(
      self,
  ) -> tuple[list[str], list[backend.Tensor], list[Any]]:
    """Helper to extract mediator names, scale factors, and decay functions."""
    mediator_names = []
    scale_factors = []
    decay_functions_list = []

    for mediator in self._mediators:
      mediator_names.append(mediator)
      scale_factors.append(
          self.model_context.get_media_scaling_factor(mediator)
      )
      channel_params = self.model_context.get_channel_parameters(mediator)
      decay_functions_list.append(channel_params.decay_spec)

    return mediator_names, scale_factors, decay_functions_list

  def _get_remaining_incremental_kpi(
      self,
      data_tensors: tensors.DataTensors,
      dist_tensors: tensors.DistributionTensors,
      non_media_treatments_baseline_normalized: Optional[
          Sequence[float]
      ] = None,
  ) -> backend.Tensor:
    """Computes remaining incremental KPI. The incremental KPI from the 2nd
    stage model (`meridian`)."""
    return self.get_incremental_kpi(
        data_tensors=data_tensors,
        dist_tensors=dist_tensors,
        non_media_treatments_baseline_normalized=non_media_treatments_baseline_normalized,
    )

  def _get_indirect_incremental_kpi(
      self,
      analyzer_mediator: analyzer.Analyzer,
      data_tensors_mediator: tensors.DataTensors,
      dist_tensors_mediator: tensors.DistributionTensors,
      dist_tensors: tensors.DistributionTensors,
      scale_factors: backend.Tensor,
      decay_functions: Any,
      has_mapping: Sequence[bool],
      channel_indices_1s: Sequence[int],
      mediator_name: str,
      include_non_paid_channels: bool,
      non_media_treatments_baseline_normalized_mediator: Optional[
          Sequence[float]
      ] = None,
  ) -> backend.Tensor:
    """Computes indirect incremental KPI through the mediator."""
    # Step 1: Stage 1 KPI contribution on original mediator scale.
    mediator_kpi = analyzer_mediator.get_incremental_kpi(
        data_tensors=data_tensors_mediator,
        dist_tensors=dist_tensors_mediator,
        non_media_treatments_baseline_normalized=non_media_treatments_baseline_normalized_mediator,
    )
    mediator_original = analyzer_mediator.inverse_outcome(
        mediator_kpi,
        use_kpi=True,
        revenue_per_kpi=data_tensors_mediator.revenue_per_kpi,
    )

    # Map Stage 1 media channels to Stage 2 media channels.
    mediator_incr_list = []
    for i, mapped in enumerate(has_mapping):
      if mapped:
        mediator_incr_list.append(mediator_original[..., channel_indices_1s[i]])
      else:
        mediator_incr_list.append(backend.zeros_like(mediator_original[..., 0]))
    indirect_kpi = backend.stack(mediator_incr_list, axis=-1)

    # Step 2: Rescale mediator effects for Stage 2
    indirect_kpi /= scale_factors[None, None, :, None, None]

    # Step 3: Apply Mediator Adstock transformation (from Stage 2 model)
    alpha_mediator = self.model_context.get_channel_parameter_tensor(
        dist_tensors,
        param_base_name="alpha",
        channel_name=mediator_name,
    )
    beta_g_mediator = self.model_context.get_channel_parameter_tensor(
        dist_tensors,
        param_base_name=constants.BETA_G,
        channel_name=mediator_name,
    )

    n_chains, n_draws_batch, n_geos, n_times, n_paid_2s = indirect_kpi.shape
    alpha_broadcast = backend.repeat(
        alpha_mediator[:, :, None], n_paid_2s, axis=-1
    )

    batch_size_total = n_chains * n_draws_batch
    indirect_kpi_reshaped = backend.reshape(
        indirect_kpi, (batch_size_total, n_geos, n_times, n_paid_2s)
    )
    alpha_broadcast_reshaped = backend.reshape(
        alpha_broadcast, (batch_size_total, n_paid_2s)
    )

    model_eqs = equations.ModelEquations(self.model_context)
    indirect_kpi_transformed = model_eqs.adstock_hill_media(
        media=indirect_kpi_reshaped,
        alpha=alpha_broadcast_reshaped,
        ec=backend.zeros_like(alpha_broadcast_reshaped),
        slope=backend.ones_like(alpha_broadcast_reshaped),
        decay_functions=decay_functions,
        saturation_spec="none",
        n_times_output=n_times,
    )

    indirect_kpi_transformed = backend.reshape(
        indirect_kpi_transformed,
        (n_chains, n_draws_batch, n_geos, n_times, n_paid_2s),
    )

    # Step 4: Multiply by Stage 2 mediator coefficient
    combined_media_kpi_indirect = backend.einsum(
        "...gtm,...g->...gtm", indirect_kpi_transformed, beta_g_mediator
    )

    # Pad indirect KPI with zeros for non-paid channels
    if include_non_paid_channels:
      n_non_paid_2s = (
          self.model_context.n_organic_media_channels
          + self.model_context.n_organic_rf_channels
          + self.model_context.n_non_media_channels
      )
      if n_non_paid_2s > 0:
        padding = backend.zeros(
            (n_chains, n_draws_batch, n_geos, n_times, n_non_paid_2s),
            dtype=backend.float_dtype,
        )
        combined_media_kpi_indirect = backend.concatenate(
            [combined_media_kpi_indirect, padding], axis=-1
        )

    return combined_media_kpi_indirect

  @backend.function(
      jit_compile=True,
      static_argnames=[
          "effect_type",
          "inverse_transform_outcome",
          "use_kpi",
          "selected_geos",
          "selected_times",
          "aggregate_geos",
          "aggregate_times",
          "include_non_paid_channels",
          "has_mappings",
          "channel_indices_1s_list",
          "mediator_names",
          "decay_functions_list",
          "analyzer_mediators_list",
      ],
  )
  def _incremental_outcome_impl(
      self,
      data_tensors: tensors.DataTensors,
      dist_tensors: tensors.DistributionTensors,
      data_tensors_mediators: Sequence[tensors.DataTensors],
      dist_tensors_mediators: Sequence[tensors.DistributionTensors],
      scale_factors: Sequence[backend.Tensor],
      decay_functions_list: Sequence[Any],
      non_media_treatments_baseline_normalized: Optional[
          Sequence[float]
      ] = None,
      non_media_treatments_baseline_normalized_mediators: Optional[
          Sequence[Optional[Sequence[float]]]
      ] = None,
      effect_type: str = "total",
      inverse_transform_outcome: bool = True,
      use_kpi: bool = False,
      selected_geos: Optional[Sequence[str]] = None,
      selected_times: Optional[Sequence[str]] = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      include_non_paid_channels: bool = True,
      has_mappings: Sequence[Sequence[bool]] = (),
      channel_indices_1s_list: Sequence[Sequence[int]] = (),
      mediator_names: Sequence[str] = (),
      analyzer_mediators_list: Sequence[analyzer.Analyzer] = (),
  ) -> backend.Tensor:
    """Computes incremental outcome (revenue or KPI) on a batch of data."""
    # 1. Remaining effect from the primary model (Stage 2)
    remaining_kpi = self._get_remaining_incremental_kpi(
        data_tensors=data_tensors,
        dist_tensors=dist_tensors,
        non_media_treatments_baseline_normalized=non_media_treatments_baseline_normalized,
    )

    # 2. Indirect effects through mediators
    total_indirect_kpi = backend.zeros_like(remaining_kpi)
    if effect_type in ("total", "indirect"):
      for i in range(len(mediator_names)):
        non_media_baseline_med = (
            non_media_treatments_baseline_normalized_mediators[i]
            if non_media_treatments_baseline_normalized_mediators is not None
            else None
        )
        indirect_kpi = self._get_indirect_incremental_kpi(
            analyzer_mediator=analyzer_mediators_list[i],
            data_tensors_mediator=data_tensors_mediators[i],
            dist_tensors_mediator=dist_tensors_mediators[i],
            dist_tensors=dist_tensors,
            scale_factors=scale_factors[i],
            decay_functions=decay_functions_list[i],
            has_mapping=has_mappings[i],
            channel_indices_1s=channel_indices_1s_list[i],
            mediator_name=mediator_names[i],
            include_non_paid_channels=include_non_paid_channels,
            non_media_treatments_baseline_normalized_mediator=(
                non_media_baseline_med
            ),
        )
        total_indirect_kpi += indirect_kpi
    # 3. Branch based on effect_type
    if effect_type == "indirect":
      total_transformed_outcome = total_indirect_kpi
    elif effect_type == "remaining":
      total_transformed_outcome = remaining_kpi
    elif effect_type == "total":
      total_transformed_outcome = total_indirect_kpi + remaining_kpi
    else:
      raise ValueError(
          f"Invalid effect_type '{effect_type}'. Expected 'total',"
          " 'indirect', or 'remaining'."
      )

    # 4. Final Inverse Transformation and Aggregation
    if inverse_transform_outcome:
      incremental_outcome = self.inverse_outcome(
          total_transformed_outcome,
          use_kpi=use_kpi,
          revenue_per_kpi=data_tensors.revenue_per_kpi,
      )
    else:
      incremental_outcome = total_transformed_outcome

    # Resolve actual indices using the builder for final aggregation
    inputs_for_indices = tensors.DataTensorsBuilder(
        self.model_context
    ).build_unscaled_inputs(
        selected_geos=selected_geos, selected_times=selected_times
    )

    return self.filter_and_aggregate_by_indices(
        tensor=incremental_outcome,
        geo_indices=inputs_for_indices.geo_indices,
        time_indices=inputs_for_indices.time_indices,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        flexible_time_dim=True,
        has_media_dim=True,
    )

  def incremental_outcome(
      self,
      use_posterior: bool = True,
      new_data: Optional[tensors.DataTensors] = None,
      non_media_baseline_values: Optional[Sequence[float]] = None,
      scaling_factor0: float = 0.0,
      scaling_factor1: float = 1.0,
      selected_geos: Optional[Sequence[str]] = None,
      selected_times: Optional[Sequence[str]] = None,
      media_selected_times: Optional[Sequence[str]] = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      inverse_transform_outcome: bool = True,
      use_kpi: bool = False,
      by_reach: bool = True,
      include_non_paid_channels: bool = True,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
      effect_type: str = "total",
      *,
      non_media_baseline_values_mediator: Optional[
          Sequence[float] | Mapping[str, Sequence[float]]
      ] = None,
  ) -> backend.Tensor:
    """Calculates either the posterior or prior total incremental outcome."""
    if effect_type not in ("total", "indirect", "remaining"):
      raise ValueError(
          f"Invalid effect_type '{effect_type}'. Expected 'total',"
          " 'indirect', or 'remaining'."
      )

    m_context = self.model_context
    use_kpi = self._use_kpi(use_kpi)
    self._check_kpi_transformation(inverse_transform_outcome, use_kpi)
    if m_context.is_national:
      analyzer._warn_if_geo_arg_in_kwargs(
          aggregate_geos=aggregate_geos,
          selected_geos=selected_geos,
      )

    dist_type = constants.POSTERIOR if use_posterior else constants.PRIOR
    if dist_type not in self.inference_data.groups():
      raise analyzer.errors.NotFittedModelError(
          f"sample_{dist_type}() must be called prior to calling this method."
      )
    for mediator, analyzer_mediator in self._analyzer_mediators.items():
      if dist_type not in analyzer_mediator.inference_data.groups():
        raise analyzer.errors.NotFittedModelError(
            f"sample_{dist_type}() must be called for the mediator model"
            f" '{mediator}'."
        )

    if scaling_factor1 <= scaling_factor0 or scaling_factor0 < 0:
      raise ValueError(
          "Invalid scaling factors. Ensure 0 <= scaling_factor0 <"
          " scaling_factor1."
      )

    # 1. Build inputs for Stage 2
    builder = tensors.DataTensorsBuilder(self.model_context)
    inputs0 = builder.build_counterfactual_inputs(
        new_data=new_data,
        scaling_factor=scaling_factor0,
        non_media_baseline_values=non_media_baseline_values,
        selected_geos=selected_geos,
        selected_times=selected_times,
        media_selected_times=media_selected_times,
        by_reach=by_reach,
        include_non_paid_channels=include_non_paid_channels,
        is_baseline=True,
    )
    inputs1 = builder.build_counterfactual_inputs(
        new_data=new_data,
        scaling_factor=scaling_factor1,
        non_media_baseline_values=non_media_baseline_values,
        selected_geos=selected_geos,
        selected_times=selected_times,
        media_selected_times=media_selected_times,
        by_reach=by_reach,
        include_non_paid_channels=include_non_paid_channels,
        is_baseline=False,
    )

    data_tensors0 = dataclasses.replace(inputs0.tensors, time=None)
    data_tensors1 = dataclasses.replace(inputs1.tensors, time=None)

    media_channels_2s = list(
        self.model_context.input_data.get_all_paid_channels()
    )
    n_total_media_2s = len(media_channels_2s)

    # Pre-fetch unified mediator metadata
    mediator_names, scale_factors, decay_functions_list = (
        self._get_mediator_metadata()
    )

    has_mappings = []
    channel_indices_1s_list = []
    analyzer_mediators_list = []
    data_tensors0_mediators = []
    data_tensors1_mediators = []
    non_media_treatments_baseline_normalized_mediators = []

    # 2. Build inputs and channel maps for all Stage 1 Mediators
    for mediator in self._mediators:
      mediator_model = self._mediator_models[mediator]
      analyzer_mediator = self._analyzer_mediators[mediator]
      analyzer_mediators_list.append(analyzer_mediator)

      media_channels_1s = list(
          mediator_model.model_context.input_data.get_all_paid_channels()
      )
      media_channel_map = {
          i: media_channels_1s.index(ch)
          for i, ch in enumerate(media_channels_2s)
          if ch in media_channels_1s
      }

      channel_indices_1s = []
      has_mapping = []
      for i in range(n_total_media_2s):
        if i in media_channel_map:
          channel_indices_1s.append(media_channel_map[i])
          has_mapping.append(True)
        else:
          channel_indices_1s.append(0)
          has_mapping.append(False)

      has_mappings.append(tuple(has_mapping))
      channel_indices_1s_list.append(tuple(channel_indices_1s))

      baseline_vals_med = None
      if non_media_baseline_values_mediator is not None:
        if isinstance(non_media_baseline_values_mediator, dict):
          baseline_vals_med = non_media_baseline_values_mediator.get(mediator)
        elif isinstance(non_media_baseline_values_mediator, (list, tuple)):
          if len(non_media_baseline_values_mediator) == len(self._mediators):
            baseline_vals_med = [
                non_media_baseline_values_mediator[
                    self._mediators.index(mediator)
                ]
            ]
          else:
            baseline_vals_med = non_media_baseline_values_mediator

      builder_med = tensors.DataTensorsBuilder(mediator_model.model_context)

      mapped_new_data = _map_new_data_to_mediator(
          new_data, self.model_context, mediator_model.model_context
      )

      inputs0_med = builder_med.build_counterfactual_inputs(
          new_data=mapped_new_data,
          scaling_factor=scaling_factor0,
          non_media_baseline_values=baseline_vals_med,
          selected_geos=selected_geos,
          selected_times=selected_times,
          media_selected_times=media_selected_times,
          by_reach=by_reach,
          include_non_paid_channels=include_non_paid_channels,
          is_baseline=True,
      )
      inputs1_med = builder_med.build_counterfactual_inputs(
          new_data=mapped_new_data,
          scaling_factor=scaling_factor1,
          non_media_baseline_values=baseline_vals_med,
          selected_geos=selected_geos,
          selected_times=selected_times,
          media_selected_times=media_selected_times,
          by_reach=by_reach,
          include_non_paid_channels=include_non_paid_channels,
          is_baseline=False,
      )

      data_tensors0_mediators.append(
          dataclasses.replace(inputs0_med.tensors, time=None)
      )
      data_tensors1_mediators.append(
          dataclasses.replace(inputs1_med.tensors, time=None)
      )
      non_media_treatments_baseline_normalized_mediators.append(
          inputs1_med.non_media_baseline_normalized
      )

    # We must always include non-paid channels for Stage 2 to ensure organic
    # mediator parameters (alpha_om, beta_gom) are available for XLA indirect
    # calcs.
    param_list = self._get_causal_param_names(include_non_paid_channels=True)

    dim_kwargs = {
        "selected_geos": (
            tuple(selected_geos) if selected_geos is not None else None
        ),
        "selected_times": (
            tuple(selected_times) if selected_times is not None else None
        ),
        "aggregate_geos": aggregate_geos,
        "aggregate_times": aggregate_times,
    }
    mapping_kwargs = {
        "mediator_names": tuple(mediator_names),
        "has_mappings": tuple(has_mappings),
        "channel_indices_1s_list": tuple(channel_indices_1s_list),
        "decay_functions_list": tuple(decay_functions_list),
        "analyzer_mediators_list": tuple(analyzer_mediators_list),
    }
    incremental_outcome_kwargs = {
        "effect_type": effect_type,
        "inverse_transform_outcome": inverse_transform_outcome,
        "use_kpi": use_kpi,
        "include_non_paid_channels": include_non_paid_channels,
        "non_media_treatments_baseline_normalized": (
            inputs1.non_media_baseline_normalized
        ),
        "non_media_treatments_baseline_normalized_mediators": tuple(
            non_media_treatments_baseline_normalized_mediators
        ),
        **dim_kwargs,
        **mapping_kwargs,
    }

    # 3. Calculate metrics sequentially in batches
    stage2_gen = self.yield_batched_distribution_tensors(
        param_list, use_posterior=use_posterior, batch_size=batch_size
    )
    mediator_gens = []
    for mediator in self._mediators:
      analyzer_med = self._analyzer_mediators[mediator]
      param_list_med = analyzer_med._get_causal_param_names(
          include_non_paid_channels=include_non_paid_channels
      )
      mediator_gens.append(
          analyzer_med.yield_batched_distribution_tensors(
              param_list_med, use_posterior=use_posterior, batch_size=batch_size
          )
      )

    incremental_outcome_temps = []

    for dist_tensors, *dist_tensors_mediators in zip(
        stage2_gen, *mediator_gens
    ):
      batch_incr = self._incremental_outcome_impl(
          data_tensors=data_tensors1,
          dist_tensors=dist_tensors,
          data_tensors_mediators=tuple(data_tensors1_mediators),
          dist_tensors_mediators=tuple(dist_tensors_mediators),
          scale_factors=tuple(scale_factors),
          **incremental_outcome_kwargs,
      )

      if scaling_factor0 != 0 or (
          inputs0.media_selected_times_mask is not None
          and not all(inputs0.media_selected_times_mask)
      ):
        kwargs_0 = incremental_outcome_kwargs.copy()
        kwargs_0["non_media_treatments_baseline_normalized"] = (
            inputs0.non_media_baseline_normalized
        )
        batch_incr -= self._incremental_outcome_impl(
            data_tensors=data_tensors0,
            dist_tensors=dist_tensors,
            data_tensors_mediators=tuple(data_tensors0_mediators),
            dist_tensors_mediators=tuple(dist_tensors_mediators),
            scale_factors=tuple(scale_factors),
            **kwargs_0,
        )
      incremental_outcome_temps.append(batch_incr)

    return backend.concatenate(incremental_outcome_temps, axis=1)

  @backend.function(
      jit_compile=True,
      static_argnames=[
          "mediator_names",
          "decay_functions_list",
      ],
  )
  def _expected_outcome_impl(
      self,
      data_tensors_no_mediators: tensors.DataTensors,
      dist_tensors: tensors.DistributionTensors,
      mediator_expected_batches: Sequence[backend.Tensor],
      scale_factors: Sequence[backend.Tensor],
      mediator_names: Sequence[str],
      decay_functions_list: Sequence[Any],
  ) -> backend.Tensor:
    """Computes expected outcome for a batch of draws."""
    # 1. Base KPI means (excluding mediator)
    kpi_means_base = self.get_kpi_means(
        data_tensors=data_tensors_no_mediators,
        dist_tensors=dist_tensors,
    )
    total_mediator_effect = backend.zeros_like(kpi_means_base)

    # 2. Loop over mediators to compute their effects
    for i, mediator_name in enumerate(mediator_names):
      mediator_expected_batch = mediator_expected_batches[i]
      scale_factor = scale_factors[i]
      decay_functions = decay_functions_list[i]

      # Scale mediator draws to Stage 2 scale.
      mediator_scaled = (
          mediator_expected_batch / scale_factor[None, None, :, None]
      )
      n_chains, n_draws_batch, n_geos, n_times = mediator_scaled.shape

      batch_size_total = n_chains * n_draws_batch
      mediator_reshaped = backend.reshape(
          mediator_scaled, (batch_size_total, n_geos, n_times, 1)
      )

      alpha_mediator = self.model_context.get_channel_parameter_tensor(
          dist_tensors,
          param_base_name="alpha",
          channel_name=mediator_name,
      )
      alpha_reshaped = backend.reshape(alpha_mediator, (batch_size_total, 1))

      model_eqs = equations.ModelEquations(self.model_context)
      transformed_mediator = model_eqs.adstock_hill_media(
          media=mediator_reshaped,
          alpha=alpha_reshaped,
          ec=backend.zeros_like(alpha_reshaped),
          slope=backend.ones_like(alpha_reshaped),
          decay_functions=decay_functions,
          saturation_spec="none",
          n_times_output=n_times,
      )
      transformed_mediator = backend.reshape(
          transformed_mediator,
          (n_chains, n_draws_batch, n_geos, n_times),
      )

      beta_g_mediator = self.model_context.get_channel_parameter_tensor(
          dist_tensors,
          param_base_name=constants.BETA_G,
          channel_name=mediator_name,
      )

      mediator_effect = transformed_mediator * beta_g_mediator[..., None]
      total_mediator_effect += mediator_effect

    return kpi_means_base + total_mediator_effect

  def expected_outcome(
      self,
      use_posterior: bool = True,
      new_data: Optional[tensors.DataTensors] = None,
      selected_geos: Optional[Sequence[str]] = None,
      selected_times: Optional[Sequence[str]] = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      inverse_transform_outcome: bool = True,
      use_kpi: bool = False,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> backend.Tensor:
    """Calculates either prior or posterior expected outcome."""
    use_kpi = self._use_kpi(use_kpi)
    self._check_kpi_transformation(inverse_transform_outcome, use_kpi)
    if self.model_context.is_national:
      analyzer._warn_if_geo_arg_in_kwargs(
          aggregate_geos=aggregate_geos,
          selected_geos=selected_geos,
      )

    dist_type = constants.POSTERIOR if use_posterior else constants.PRIOR
    if dist_type not in self.inference_data.groups():
      raise analyzer.errors.NotFittedModelError(
          f"sample_{dist_type}() must be called prior to calling this method."
      )

    for mediator, analyzer_mediator in self._analyzer_mediators.items():
      if dist_type not in analyzer_mediator.inference_data.groups():
        raise analyzer.errors.NotFittedModelError(
            f"sample_{dist_type}() must be called for the mediator model"
            f" '{mediator}'."
        )

    builder = tensors.DataTensorsBuilder(self.model_context)
    inputs = builder.build_scaled_inputs(
        new_data=new_data,
        include_non_paid_channels=True,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    data_tensors = dataclasses.replace(inputs.tensors, time=None)

    mediator_names, scale_factors, decay_functions_list = (
        self._get_mediator_metadata()
    )

    organic_media = data_tensors.organic_media
    assert organic_media is not None

    # Zero out the mediators in the Stage 2 organic media tensor
    mask_np = np.ones(organic_media.shape[-1], dtype=np.float32)
    for mediator in self._mediators:
      idx = self.model_context.get_channel_parameters(mediator).index
      mask_np[idx] = 0.0
    mask = backend.to_tensor(mask_np, dtype=backend.float_dtype)

    organic_media_no_mediators = organic_media * mask
    data_tensors_no_mediators = dataclasses.replace(
        data_tensors, organic_media=organic_media_no_mediators
    )

    # Pre-compute Stage 1 expected outcomes on original scale
    mediator_expecteds = []
    for mediator in self._mediators:
      analyzer_mediator = self._analyzer_mediators[mediator]
      mapped_new_data = _map_new_data_to_mediator(
          new_data, self.model_context, analyzer_mediator.model_context
      )
      mediator_expected = analyzer_mediator.expected_outcome(
          use_posterior=use_posterior,
          new_data=mapped_new_data,
          inverse_transform_outcome=True,
          use_kpi=True,
          batch_size=batch_size,
          aggregate_geos=False,
          aggregate_times=False,
      )
      mediator_expecteds.append(mediator_expected)

    param_list = (
        [constants.MU_T, constants.TAU_G]
        + ([constants.GAMMA_GC] if self.model_context.n_controls else [])
        + self._get_causal_param_names(include_non_paid_channels=True)
    )

    stage2_gen = self.yield_batched_distribution_tensors(
        param_list, use_posterior=use_posterior, batch_size=batch_size
    )

    outcome_means_temps = []

    params = (
        self.inference_data.posterior  # pyrefly: ignore[missing-attribute]
        if use_posterior
        else self.inference_data.prior  # pyrefly: ignore[missing-attribute]
    )
    n_draws = params.draw.size
    batch_starting_indices = np.arange(n_draws, step=batch_size)

    for start_index, dist_tensors in zip(batch_starting_indices, stage2_gen):
      stop_index = np.min([n_draws, start_index + batch_size])
      mediator_expected_batches = [
          med_exp[:, start_index:stop_index, ...]
          for med_exp in mediator_expecteds
      ]

      outcome_means_temps.append(
          self._expected_outcome_impl(
              data_tensors_no_mediators=data_tensors_no_mediators,
              dist_tensors=dist_tensors,
              mediator_expected_batches=tuple(mediator_expected_batches),
              scale_factors=tuple(scale_factors),
              mediator_names=tuple(mediator_names),
              decay_functions_list=tuple(decay_functions_list),
          )
      )

    outcome_means = backend.concatenate(outcome_means_temps, axis=1)

    if inverse_transform_outcome:
      outcome_means = self.model_context.kpi_transformer.inverse(outcome_means)
      if not use_kpi:
        revenue_per_kpi = (
            inputs.tensors.revenue_per_kpi
            if inputs.tensors.revenue_per_kpi is not None
            else self.model_context.revenue_per_kpi
        )
        outcome_means *= revenue_per_kpi

    return self.filter_and_aggregate_by_indices(
        outcome_means,
        geo_indices=inputs.geo_indices,
        time_indices=inputs.time_indices,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
    )


def compute_roi_decomposition(
    analyzer_full_funnel: AnalyzerFullFunnel,
    confidence_level: float = 0.8,
) -> pd.DataFrame:
  """Computes draw-level indirect, remaining, and total ROI distributions and
  summaries.

  Args:
    analyzer_full_funnel: Fitted full-funnel analyzer.
    confidence_level: Credible interval coverage level (default 0.8).

  Returns:
    A DataFrame with columns: 'channel', 'indirect_roi_mean',
    'indirect_roi_ci_lo', 'indirect_roi_ci_hi', 'remaining_roi_mean',
    'remaining_roi_ci_lo', 'remaining_roi_ci_hi', 'total_roi_mean',
    'total_roi_ci_lo', 'total_roi_ci_hi'.
  """
  channels = list(
      analyzer_full_funnel.model_context.input_data.get_all_paid_channels()
  )
  spend = analyzer_full_funnel.model_context.total_spend
  if len(spend.shape) > 1:
    channel_spend = backend.reduce_sum(
        spend,
        axis=range(len(spend.shape) - 1),
    )
  else:
    channel_spend = spend

  indirect_outcome = analyzer_full_funnel.incremental_outcome(
      effect_type="indirect",
      include_non_paid_channels=False,
  )
  remaining_outcome = analyzer_full_funnel.incremental_outcome(
      effect_type="remaining",
      include_non_paid_channels=False,
  )

  # Draw-level ROI tensors
  indirect_roi = indirect_outcome / channel_spend
  remaining_roi = remaining_outcome / channel_spend
  total_roi = indirect_roi + remaining_roi

  indirect_stats = analyzer.get_central_tendency_and_ci(
      indirect_roi, confidence_level=confidence_level
  )
  remaining_stats = analyzer.get_central_tendency_and_ci(
      remaining_roi, confidence_level=confidence_level
  )
  total_stats = analyzer.get_central_tendency_and_ci(
      total_roi, confidence_level=confidence_level
  )

  return pd.DataFrame(
      {
          "channel": channels,
          "indirect_roi_mean": indirect_stats[:, 0],
          "indirect_roi_ci_lo": indirect_stats[:, 1],
          "indirect_roi_ci_hi": indirect_stats[:, 2],
          "remaining_roi_mean": remaining_stats[:, 0],
          "remaining_roi_ci_lo": remaining_stats[:, 1],
          "remaining_roi_ci_hi": remaining_stats[:, 2],
          "total_roi_mean": total_stats[:, 0],
          "total_roi_ci_lo": total_stats[:, 1],
          "total_roi_ci_hi": total_stats[:, 2],
      }
  )


def plot_roi_decomposition(
    analyzer_full_funnel: AnalyzerFullFunnel,
    confidence_level: float = 0.8,
) -> alt.Chart:
  """Plots a stacked ROI bar chart (indirect vs remaining) per channel."""
  ci = int(confidence_level * 100)
  title = f"ROI by channel with {ci}% credible interval"

  roi_df = compute_roi_decomposition(analyzer_full_funnel, confidence_level)
  bars_data = []
  for _, row in roi_df.iterrows():
    bars_data.append(
        {
            "channel": row["channel"],
            "component": "Indirect ROI",
            "roi": row["indirect_roi_mean"],
            "order": 1,
        }
    )
    bars_data.append(
        {
            "channel": row["channel"],
            "component": "Remaining ROI",
            "roi": row["remaining_roi_mean"],
            "order": 2,
        }
    )
  bars_df = pd.DataFrame(bars_data)

  color_scale = alt.Scale(
      domain=["Total ROI", "Indirect ROI", "Remaining ROI"],
      range=[constants.GREY_800, constants.YELLOW_600, constants.BLUE_700],
  )

  bars = (
      alt.Chart(bars_df)
      .mark_bar(
          size=constants.BAR_SIZE,
          cornerRadiusEnd=constants.CORNER_RADIUS,
      )
      .encode(
          x=alt.X(
              "channel:N",
              title="Channel",
              axis=alt.Axis(labelAngle=-45),
          ),
          y=alt.Y(
              "roi:Q",
              title="ROI",
              stack="zero",
              axis=alt.Axis(gridDash=[3, 2], **formatter.Y_AXIS_TITLE_CONFIG),
          ),
          color=alt.Color(
              "component:N",
              scale=color_scale,
              legend=alt.Legend(
                  title=None,
                  orient="bottom",
                  direction="horizontal",
              ),
          ),
          order=alt.Order("order:Q"),
          tooltip=[
              alt.Tooltip("channel:N", title="Channel"),
              alt.Tooltip("component:N", title="Component"),
              alt.Tooltip("roi:Q", title="Component Mean ROI", format=".2f"),
          ],
      )
      .properties(width=alt.Step(80))
  )

  bar_width = 2
  error_bar = (
      alt.Chart(roi_df)
      .mark_errorbar(ticks={"size": 12}, color=constants.GREY_800)
      .encode(
          x=alt.X("channel:N"),
          y=alt.Y("total_roi_ci_hi:Q", title="ROI"),
          y2=alt.Y2("total_roi_ci_lo:Q"),
          strokeWidth=alt.value(bar_width),
      )
  )

  mean_dot = (
      alt.Chart(roi_df)
      .transform_calculate(component="'Total ROI'")
      .mark_point(
          filled=True,
          size=constants.POINT_SIZE,
          color=constants.GREY_800,
          tooltip=True,
      )
      .encode(
          x=alt.X("channel:N"),
          y=alt.Y("total_roi_mean:Q"),
          color=alt.Color("component:N", scale=color_scale),
          tooltip=[
              alt.Tooltip("channel:N", title="Channel"),
              alt.Tooltip(
                  "total_roi_mean:Q", title="Total ROI Mean", format=".2f"
              ),
              alt.Tooltip(
                  "total_roi_ci_lo:Q",
                  title=f"Total ROI {ci}% CI Lower",
                  format=".2f",
              ),
              alt.Tooltip(
                  "total_roi_ci_hi:Q",
                  title=f"Total ROI {ci}% CI Upper",
                  format=".2f",
              ),
          ],
      )
  )

  plot = alt.layer(bars, error_bar, mean_dot)
  return (
      plot.configure_tick(bandSize=10, thickness=bar_width)
      .properties(title=formatter.custom_title_params(title))
      .configure_axis(
          titlePadding=constants.PADDING_10, **formatter.TEXT_CONFIG
      )
  )


def _create_media_pulse_data(
    analyzer_full_funnel: AnalyzerFullFunnel,
    pulse_fraction: float = 0.01,
) -> tensors.DataTensors:
  """Creates a DataTensors container simulating an impulse at time unit 0."""
  input_data = analyzer_full_funnel.model_context.input_data

  pulse_kwargs = {}

  if input_data.media is not None:
    hist_media = input_data.media.values
    weekly_avg = np.mean(hist_media, axis=1)
    pulse_media = np.zeros_like(hist_media)
    pulse_media[:, 0, :] = pulse_fraction * weekly_avg
    pulse_kwargs["media"] = pulse_media

  if input_data.organic_media is not None:
    hist_organic = input_data.organic_media.values
    weekly_avg_om = np.mean(hist_organic, axis=1)
    pulse_organic = np.zeros_like(hist_organic)
    pulse_organic[:, 0, :] = pulse_fraction * weekly_avg_om
    pulse_kwargs["organic_media"] = pulse_organic

  if input_data.reach is not None and input_data.frequency is not None:
    hist_reach = input_data.reach.values
    weekly_avg_reach = np.mean(hist_reach, axis=1)
    pulse_reach = np.zeros_like(hist_reach)
    pulse_reach[:, 0, :] = pulse_fraction * weekly_avg_reach
    pulse_kwargs["reach"] = pulse_reach
    avg_freq = np.mean(input_data.frequency.values, axis=1, keepdims=True)
    pulse_kwargs["frequency"] = np.broadcast_to(
        avg_freq, input_data.frequency.shape
    )

  if (
      input_data.organic_reach is not None
      and input_data.organic_frequency is not None
  ):
    hist_orf = input_data.organic_reach.values
    weekly_avg_orf = np.mean(hist_orf, axis=1)
    pulse_orf = np.zeros_like(hist_orf)
    pulse_orf[:, 0, :] = pulse_fraction * weekly_avg_orf
    pulse_kwargs["organic_reach"] = pulse_orf
    avg_orf_freq = np.mean(
        input_data.organic_frequency.values, axis=1, keepdims=True
    )
    pulse_kwargs["organic_frequency"] = np.broadcast_to(
        avg_orf_freq, input_data.organic_frequency.shape
    )

  return tensors.DataTensors(**pulse_kwargs)


def plot_compounding_adstock_decay(
    analyzer_full_funnel: AnalyzerFullFunnel,
    confidence_level: float = 0.8,
    use_posterior: bool = True,
) -> alt.Chart:
  """Plots compounding adstock decay curves across indirect, remaining, and
  total pathways."""
  # 1. Dynamic lag horizon
  remaining_max_lag = analyzer_full_funnel.model_context.model_spec.max_lag
  compounding_max_lag = remaining_max_lag

  mediator_model_lags = [
      m.model_context.model_spec.max_lag
      for m in analyzer_full_funnel._mediator_models.values()
  ]
  compounding_max_lag += max(mediator_model_lags)

  # 2. Media pulse simulation
  pulse_data = _create_media_pulse_data(analyzer_full_funnel)

  # 3. Counterfactual evaluation
  indirect_outcome = analyzer_full_funnel.incremental_outcome(
      new_data=pulse_data,
      aggregate_geos=True,
      aggregate_times=False,
      use_kpi=True,
      effect_type="indirect",
      include_non_paid_channels=True,
      use_posterior=use_posterior,
  )
  remaining_outcome = analyzer_full_funnel.incremental_outcome(
      new_data=pulse_data,
      aggregate_geos=True,
      aggregate_times=False,
      use_kpi=True,
      effect_type="remaining",
      include_non_paid_channels=True,
      use_posterior=use_posterior,
  )

  # Filter to within horizon
  indirect_horizon = indirect_outcome[..., : compounding_max_lag + 1, :]
  remaining_horizon = remaining_outcome[..., : compounding_max_lag + 1, :]
  total_horizon = indirect_horizon + remaining_horizon

  # 4. Draw-wise cumulative sum normalization
  total_sum = np.sum(total_horizon, axis=2, keepdims=True)
  total_sum_safe = np.where(total_sum == 0, 1.0, total_sum)

  norm_indirect = indirect_horizon / total_sum_safe
  norm_remaining = remaining_horizon / total_sum_safe
  norm_total = total_horizon / total_sum_safe

  # 5. Extract statistics
  indirect_stats = analyzer.get_central_tendency_and_ci(
      norm_indirect, confidence_level=confidence_level, axis=(0, 1)
  )
  remaining_stats = analyzer.get_central_tendency_and_ci(
      norm_remaining, confidence_level=confidence_level, axis=(0, 1)
  )
  total_stats = analyzer.get_central_tendency_and_ci(
      norm_total, confidence_level=confidence_level, axis=(0, 1)
  )

  # 6. Channels to display
  all_channels = list(
      analyzer_full_funnel.model_context.input_data.get_all_channels()
  )
  mediators = set(analyzer_full_funnel._mediators)
  mediated_channels = set()
  for m in analyzer_full_funnel._mediator_models.values():
    mediated_channels.update(m.model_context.input_data.get_all_paid_channels())

  channels_to_plot = [ch for ch in all_channels if ch not in mediators]

  # 7. Construct plotting DataFrame
  records = []
  for ch in channels_to_plot:
    ch_idx = all_channels.index(ch)
    is_mediated = ch in mediated_channels
    channel_label = (
        f"{ch} (Stage 1 and 2)" if is_mediated else f"{ch} (Stage 2 only)"
    )
    max_lag = compounding_max_lag if is_mediated else remaining_max_lag
    for t in range(max_lag + 1):
      if is_mediated:
        records.append(
            {
                "channel": channel_label,
                "time_units": t,
                "component": "Indirect Adstock",
                "mean": float(indirect_stats[t, ch_idx, 0]),
                "ci_lo": float(indirect_stats[t, ch_idx, 1]),
                "ci_hi": float(indirect_stats[t, ch_idx, 2]),
            }
        )
      if t <= remaining_max_lag:
        records.append(
            {
                "channel": channel_label,
                "time_units": t,
                "component": "Remaining Adstock",
                "mean": float(remaining_stats[t, ch_idx, 0]),
                "ci_lo": float(remaining_stats[t, ch_idx, 1]),
                "ci_hi": float(remaining_stats[t, ch_idx, 2]),
            }
        )
      records.append(
          {
              "channel": channel_label,
              "time_units": t,
              "component": "Total Adstock",
              "mean": float(total_stats[t, ch_idx, 0]),
              "ci_lo": float(total_stats[t, ch_idx, 1]),
              "ci_hi": float(total_stats[t, ch_idx, 2]),
          }
      )

  df = pd.DataFrame(records)

  base = alt.Chart(df, width=constants.VEGALITE_FACET_DEFAULT_WIDTH)
  color_scale = alt.Scale(
      domain=[
          "Total Adstock",
          "Indirect Adstock",
          "Remaining Adstock",
      ],
      range=[constants.GREY_800, constants.YELLOW_600, constants.BLUE_700],
  )
  dash_scale = alt.Scale(
      domain=[
          "Indirect Adstock",
          "Remaining Adstock",
      ],
      range=[[4, 3], [0, 0]],
  )
  legend_cfg = alt.Legend(
      title=None,
      orient="bottom",
      direction="horizontal",
      symbolOpacity=1.0,
      symbolStrokeWidth=3,
  )

  # 1. Shaded CI band for Total Adstock (bottom layer)
  band = (
      base.transform_filter(alt.datum.component == "Total Adstock")
      .mark_area(opacity=0.15, color=constants.GREY_800)
      .encode(
          x=alt.X("time_units:Q"),
          y=alt.Y(
              "ci_lo:Q",
              title="Percentage of Total Effect",
              axis=alt.Axis(format=".1%"),
          ),
          y2=alt.Y2("ci_hi:Q"),
      )
  )

  # 2. Component lines (Indirect & Remaining) rendered underneath Total line
  component_lines = (
      base.transform_filter(alt.datum.component != "Total Adstock")
      .mark_line(strokeWidth=1.5)
      .encode(
          x=alt.X("time_units:Q", title="Time Units"),
          y=alt.Y(
              "mean:Q",
              title="Percentage of Total Effect",
              axis=alt.Axis(format=".1%"),
          ),
          color=alt.Color(
              "component:N",
              scale=color_scale,
              legend=legend_cfg,
          ),
          strokeDash=alt.StrokeDash(
              "component:N",
              scale=dash_scale,
              legend=None,
          ),
      )
  )

  # 3. Total Adstock line rendered ON TOP of component lines
  total_line = (
      base.transform_filter(alt.datum.component == "Total Adstock")
      .mark_line(strokeWidth=2.5)
      .encode(
          x=alt.X("time_units:Q", title="Time Units"),
          y=alt.Y(
              "mean:Q",
              title="Percentage of Total Effect",
              axis=alt.Axis(format=".1%"),
          ),
          color=alt.Color(
              "component:N",
              scale=color_scale,
              legend=legend_cfg,
          ),
      )
  )

  # 4. Observation points with hover tooltips
  points = base.mark_circle(
      filled=True, size=constants.POINT_SIZE, opacity=1
  ).encode(
      tooltip=[
          "channel:N",
          "component:N",
          "time_units:Q",
          alt.Tooltip("mean:Q", title="Mean Effect", format=".2%"),
          alt.Tooltip("ci_lo:Q", title="Credible Interval Lower", format=".2%"),
          alt.Tooltip("ci_hi:Q", title="Credible Interval Upper", format=".2%"),
      ],
      x=alt.X("time_units:Q"),
      y=alt.Y("mean:Q"),
      color=alt.Color(
          "component:N",
          scale=color_scale,
          legend=legend_cfg,
      ),
  )

  plot = alt.layer(band, component_lines, total_line, points)

  return (
      plot.facet("channel:N", columns=3)
      .properties(
          title=formatter.custom_title_params("Compounded Adstock Decay")
      )
      .configure_axis(**formatter.TEXT_CONFIG)
      .resolve_scale(x=constants.INDEPENDENT, y=constants.INDEPENDENT)
  )
