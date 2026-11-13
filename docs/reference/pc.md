# Probability of collision

`compute_pc` and every Pc method. The guides explain when each applies: [2-D probability of collision](../pc-2d.md) and [3-D, Monte Carlo and maximum Pc](../pc-3d.md).

::: keepout.pc
    options:
      members:
        - compute_pc
        - computation
        - PcComputation
        - PcResult
        - Method
        - MethodChoice
        - MethodKind
        - MethodSpec
        - METHODS
        - AUTO
        - ResultKind
        - FosterOptions
        - ElrodOptions
        - Hall3dOptions
        - MonteCarloOptions
        - foster
        - elrod
        - hall3d
        - montecarlo
        - frisbee
        - dilution
        - alfano
        - require_options

## The method modules

Each method's module explains the method, its defaults and its tolerances.

::: keepout.pc.api
    options:
      heading_level: 3
      members:
        - AUTO_MC_SAMPLES
        - AUTO_MC_SEED
        - DURATION_LIMIT
        - SIGMA_EXTENT
        - TCA_OFFSET_LIMIT
        - applicability
        - encounter_duration

::: keepout.pc.foster
    options:
      heading_level: 3
      members:
        - BREAKPOINT_GAP
        - FOSTER_RTOL
        - MIN_RTOL
        - QUAD_LIMIT
        - STEP_SIGMAS
        - TAIL_SIGMAS
        - gaussian_pc

::: keepout.pc.elrod
    options:
      heading_level: 3
      members:
        - ELROD_RTOL
        - MAX_ORDER
        - START_ORDER

::: keepout.pc.hall3d
    options:
      heading_level: 3
      members:
        - HALL_RTOL
        - QUIET_DISAGREEMENT
        - hall3d_over_scan
        - is_long

::: keepout.pc.montecarlo
    options:
      heading_level: 3
      members:
        - MC_BATCH
        - MC_CONFIDENCE
        - MC_MAX_SAMPLES
        - MC_RTOL
        - NEWTON_STEPS
        - SPEED_BOUND_FACTOR
        - montecarlo_over_scan

::: keepout.pc.maxpc
    options:
      heading_level: 3
      members:
        - DILUTION_TOLERANCE
        - SEARCH_DECADES
        - SEARCH_GAIN_RTOL
        - SEARCH_XATOL
        - SEED_SCALE_FLOOR

::: keepout.pc.result
    options:
      heading_level: 3
      members:
        - RESULT_KINDS
