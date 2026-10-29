# deep_helmholtz

Deep learned deep helmholtz energies :)

If repo is (too) empty or too messy, ask owner to add stuff or clean (might need motivation and/or suggestions).

But please read the paper first:

```
@article{fleck2026deep,
  title={Deep Learning Helmholtz Energies for Thermodynamic Property Prediction},
  author={Fleck, Maximilian and Spera, Marcelle BM},
  year={2026}
}
```

If you use the PCP-SAFT parameters cite Esper et. al. (PCP-SAFT parameters of pure substances using large experimental databases) and if you use the water data cite... Ah, read the paper. The sources are there.

## Quickstart:

install [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

then run (all packages will be downloaded and saved locally)
```
uv run jupyter-lab
```
...or use vs code. Then execute
```
code .
```
and select the uv environment.


## Notebooks

### Data Notebooks

All notebooks starting with data*.ipynb are test notebooks to build data or transform data or analyze data to make it usable for model training.

#### Data Generation

- data_eos_build_full_dataset.ipynb: generate PCP-SAFT data. RUN THIS AT FIRST!!!

- data_get_gold_subset.ipynb: extract a golden dataset with physically meaningful PCP-SAFT parameters.

- data_splitter.ipynb: splits data into train, val, test.

- data_transform_SAFT_parameters.ipynb: analyze and transform PC-SAFT parameters. RUN THIS AT SECOND!!!

- data_widen_model.ipynb: change first layer of a model to fine-tune it for a different molecule representation.

### Test Notebooks

All notebooks starting with test*.ipynb are test notebooks to test trained models on vapor-liquid equilibria.

There is one set of notebooks for models based on synthetic PC-SAFT data:

#### PC-SAFT Tests:

- test_model_base.ipynb: simple test notebook to test model for a list of molecules. RUN THIS AT 3RD!!!

- test_model_extrapolation.ipynb: tests extrapolation for unknown virtual molecules.

- test_model_optimize_inputs.ipynb: optimizes model input parameters on VLE data.


#### Water Tests:

- test_model_water_base.ipynb: simple test notebook to test the water model. RUN THIS AT 4TH!!!

- test_model_optimize_inputs.ipynb: optimizes model input parameters of the water model on water VLE data.


## Config Files

Configuration files for training run. If you use ray, some of the parameters in the confids will be overwritten. Not all parameters in the config might have effects, some are probably artefacts or point to future ideas. So be careful and check yourself if the parameter has any effects in the codebase.

## Orchestrator Scripts

Use those scripts to run training runs...

Note: fine-tuning on real-world experimental data (water example) is tricky because of overfitting. Save some intermediate model states and use the test notebooks to compare them. Pick the best...