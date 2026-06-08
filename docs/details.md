# Run Details

This file keeps run-specific instructions out of the README.

## 1. Optical Computing

Set the training mode in `tasks/param/param_onn.py`:

```python
settings['optimizer'] = 'mfo'  # model-free optimization
settings['optimizer'] = 'sbt'  # simulator-based training
settings['optimizer'] = 'hbt'  # hybrid training
settings['train_or_test'] = 'train'
```

Run training:

```bash
python tasks/main_onn.py
```

## 2. Two-Layer Simulator

The default optical computing simulator is one-layer. To use the two-layer optical computing simulator, set this in `tasks/param/param_onn.py`:

```python
settings['num_optical_computing_layers'] = 2
```

The two-layer simulator uses:

- `optics_param['optical_computing_layer1']`
- `optics_param['optical_computing_layer2']`
- `optics_param['propogator.IC1']`
- `optics_param['propogator.C1C2']`
- `optics_param['propogator.C2O']`

Run it with the same ONN entrypoint:

```bash
python tasks/main_onn.py
```

## 3. Optical Computing Test

Set testing parameters in `tasks/param/param_onn.py`:

```python
settings['train_or_test'] = 'test'
test_param['ckpt_dir'] = YOUR_CHECKPOINT_DIR
test_param['ckpt_num'] = None
```

Run testing:

```bash
python tasks/main_onn.py
```

## 4. Naive CGH

The CGH task in this repo uses the naive model-free policy-gradient method only. The advanced SGES and EBM variants are not included.

Set CGH parameters in `tasks/param/param_holo.py`:

```python
settings['pg_type'] = 'loo'
settings['optimizer'] = 'mfo'
```

Run CGH:

```bash
python tasks/main_cgh.py
```
