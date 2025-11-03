+++
title = 'Using optuna with an existing argument parser without changes'
date = 2020-01-26T09:02:08-03:00
draft = false
+++

This post was originally posted as a discussion on [optuna's github](https://github.com/optuna/optuna/issues/862).

I was doing some experiments and I had an existing code which handled argument parsing from command line using `argparse`. After I played with it a bit, I wanted to plug-in a hyperparameter tuning library to do a more thorough search. I wanted to change the existing code as little as possible, while still making possible to use the argument parser and being easy to maintain. 

A solution is to always parse `args` with the default values and after overwrite them with the `trials` parameters. Then, you can use it with either `args['lr']` or `args.lr`. Example:

```python
def update_args_(args, params):
  """updates args in-place"""
  dargs = vars(args)
  dargs.update(params)


def main(trials=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', type=float, default=0.01)
  args = parser.parse_args()
  if trials is not None:
    params = {'lr': trials.suggest_loguniform('lr', 1e-7, 1e2)}
    update_args_(args, params)
  optimizer = optim.SGD(model.parameters(), lr=args.lr)
  
if __name__ == '__main__':
  study = optuna.create_study(direction='maximize')
  study.optimize(main, n_trials=10)
  #main()  # uncomment to run normaly
```

This way, you still keep both options. If you just want to disable `optuna` and run with `argparse`, just comment out `main` and remove the `study` lines. Or keep another file only for hyperparameter tuning.
