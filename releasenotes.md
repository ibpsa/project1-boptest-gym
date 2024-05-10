# Release Notes

BOPTEST-Gym has two main dependencies: BOPTEST and Stable-Baselines3. For simplicity, the first two digits of the version number match the same two digits of the BOPTEST version of which BOPTEST-Gym is compatible with. For example, BOPTEST-Gym v0.3.x is compatible with BOPTEST v0.3.x. The last digit is reserved for other internal edits specific to this repository only. See [here](https://github.com/ibpsa/project1-boptest/blob/master/releasenotes.md) for BOPTEST release notes. 

## BOPTEST-Gym v0.6.0

Released on 10/05/2024.

- Update CCAI tutorial for numeric results with new Python v3.10.
- Pin Dockerfile image to linux/x86_64.
- Implement functionality and example with vectorized environment for parallel learning. This is for [#133](https://github.com/ibpsa/project1-boptest-gym/issues/133). 

## BOPTEST-Gym v0.5.0

Released on 11/11/2023.

- Update for `BOPTEST v0.5.0`. This is for [#135](https://github.com/ibpsa/project1-boptest-gym/pull/136).  
- Remove arbitrarily small offset when requesting forecasts. This is for [#127](https://github.com/ibpsa/project1-boptest-gym/issues/127). 
- Implement CI and testing using GitHub Actions. This is for [#23](https://github.com/ibpsa/project1-boptest-gym/issues/23). 

## BOPTEST-Gym v0.4.0

Released on 17/07/2023.

- Update for new BOPTEST API changes when getting forecast. This is for [BOPTEST #356](https://github.com/ibpsa/project1-boptest/issues/356).  
- Update for new BOPTEST API changes when getting results. This is for [BOPTEST #398](https://github.com/ibpsa/project1-boptest/issues/398).  
- Update to `Gym v0.26.2.` and `stable-baselines3 v2.0.0`. 
- Import Gymnasium instead of Gym. Change from `compute_reward` to `get_reward` not to fall into Stable Baseline's convention for goal environments. 
- Use `terminated` and `trunctated` outputs from Gym instead of directly `done`. Return `info` upon calling the reset method of Gym.

## BOPTEST-Gym v0.3.0

Released on 25/10/2022.

- Retrieve `'payload'` after each request call to BOPTEST. 

## BOPTEST v0.2.0

Released on 18/08/2022.

This is an initial development release.
