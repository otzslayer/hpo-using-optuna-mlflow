# HPO Using Optuna and MLflow
본 저장소는 Optuna를 이용한 Hyper-parameter Optimization (HPO)를 수행하고 MLflow를 이용해 Experiment tracking을 수행하는 내용을 담고 있습니다.



## 실행 방법

0.  Anaconda로 Python 3.8+ 이상의 가상 환경을 생성하는 것을 권장드립니다.

1.  저장소를 Clone 합니다.

```
https://github.com/otzslayer/hpo-using-optuna-mlflow
```

2.  Dependency 설치

```
pip3 install requirements.txt
```

3.  `run_hpo.py` 실행

```
python run_hpo.py --n_trials 50
```

4.  MLflow UI 실행

```
$ mlflow ui
```

5.  http://127.0.0.1:5000 접속

![](https://media.vlpt.us/images/otzslayer/post/b6787b82-b277-4672-9b40-9952d80be243/mlflow_optuna_ui.png)
