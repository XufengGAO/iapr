# env installation

## Required packages

```shell
conda install -c conda-forge jupyter
conda install -c conda-forge ipywidgets
python -m pip install numpy scipy matplotlib statsmodels
jupyter nbextension enable --py widgetsnbextension
# `python -m pip install --upgrade jupyter_client`
python -m pip install -U scikit-learn scikit-image
python -m pip install webcolors opencv-python
python -m pip install jupyterlab

# to enable toc
python -m pip install nbconvert
python -m pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
```

## ToC for notebook

- install: <https://github.com/ipython-contrib/jupyter_contrib_nbextensions>

    ```shell
    python -m pip install jupyter_contrib_nbextensions
    # conda install -c conda-forge jupyter_contrib_nbextensions
    jupyter contrib nbextension install --user
    ```

- activate **Table of Contents (2)*
