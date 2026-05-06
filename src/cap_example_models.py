import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#Desde a versão 1.4.0, RandomForestClassifier tem solução nativa para missing

import sklearn
print(sklearn.__version__)

from google.colab import drive # [pt] Conecta o Colab ao Google Drive [en]
drive.mount("/content/drive")

# Parâmetros para as funções
# [pt] Caminho do banco de dados
## [pt] Como pastas compartilhadas não aparecem em "MyDrive" por padrão, é necessário primeiro adicionar o atalho ao Drive:

data_path = "/content/drive/MyDrive/Portfolio_WL/Gains_chart/app/data/cs-training.csv"

# [pt] Features iniciais para o desenvolvimento do modelo

features = ["RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents"]

# features = ['UltPercLimit',
#             'Idade',
#             'N_Atraso30_59Dias',
#             'RendaMensal',
#             'N_EmeprestimosAbertos',
#             'N_atrasos_Ult90Dias',
#             'N_emprestimos',
#             'N_Atraso60_89Dias',
#             'N_dependentes',
#             'lnRazaoGastos']

default_rf_params = {
    "n_estimators":      [100, 200, 250, 300, 400],
    "max_depth":         [2, 3, 4, 5, 8],
    "min_samples_split": [2, 5, 10, 20],
    "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.8],
    "bootstrap":         [True, False],
    "class_weight":      [None, "balanced", "balanced_subsample"],
}

# [pt] Variável resposta
target_col = "SeriousDlqin2yrs"

def _load_data(data_path: str,
               sep: str = ","):
    """
    [pt] Carrega os dados de um arquivo baseado na sua extensão.
    [en] Loads data from a file based on its extension.

    Ars:
    data_path: str
        Caminho para o arquivo de dados.
    sep: str
        Separador de colunas no arquivo de dados para as extensões .csv e .txt.

    Returns:
        Retorna o dataframe carregado.
    """

    extension = data_path.split('.')[-1].lower()

    if extension == 'csv' or extension == 'txt':
        return pd.read_csv(data_path, sep=sep)
    elif extension == 'xlsx' or extension == 'xls':
        return pd.read_excel(data_path)

    else:
        raise ValueError(
            f"[pt] Tipo de arquivo não suportado: {extension}. Por favor, forneça um arquivo .csv, .txt, .xls ou .xlsx.\n"
            f"[en] Unsupported file type: {extension}. Please provide a .csv, .txt, .xls, or .xlsx file."
        )


def _split_data(df: pd.DataFrame,
                target_col: str,
                test_size: float = 0.2,
                random_state: int = 42,
                features: list = []):
    '''
    [pt] Divide os dados em conjuntos de treino e teste, definindo a matriz de features (X) e o vetor target (y).
    [en] Splits the data into train and test sets, defining the feature matrix (X) and target vector (y).

    Args:
    df: pd.DataFrame
        [pt] Dataframe com os dados de entrada.
        [en] Input dataframe.
    target_col: str
        [pt] Nome da variável resposta (target).
        [en] Name of the target variable.
    test_size: float
        [pt] Proporção do conjunto de teste na divisão treino/teste. O valor padrão é 0.2 (20%).
        [en] Fraction of the data to use as the test set. Default is 0.2. (20%).
    random_state: int
        [pt] Semente aleatória para reprodutibilidade. O padrão é 42.
        [en] Random seed for reproducibility. Default is 42.
    features: list
        [pt] Lista de colunas a serem utilizadas como features (variáveis preditoras).
             Se não fornecida, todas as colunas exceto target_col serão utilizadas.
        [en] List of feature columns to use as predictors.
             If not provided, all columns except target_col will be used.

    Returns:
        [pt] Retorna os conjuntos X_train, X_test, y_train, y_test.
        [en] Returns X_train, X_test, y_train, y_test splits.
    '''

    if not features:
        X = df.drop(columns=[target_col])
    else:
        X = df[features]

    y = df[target_col].astype(int)

    # [pt] Usamos `stratify` para garantir que a proporção das classes do target seja preservada nos conjuntos de treino e teste.
    # [en] `stratify` is used to ensure that class proportions in the target are preserved across train and test splits.
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def _get_features_lists(df_train: pd.DataFrame):
    """
    [pt] Separa as colunas do DataFrame em listas de features numéricas e categóricas com base nas colunas existentes.
    [en] Splits DataFrame columns into lists of numerical and categorical features based on existing columns.

    Args:
    df_train: pd.DataFrame
        [pt] Conjunto de treino utilizado para identificar os tipos de cada coluna.
        [en] Training set used to identify column types.

    Returns:
        [pt] Tupla contendo (num_cols, cat_cols): listas de colunas numéricas e categóricas, respectivamente.
        [en] Tuple containing (num_cols, cat_cols): lists of numerical and categorical columns, respectively.
    """

    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_train.select_dtypes(exclude=[np.number]).columns.tolist()

    return num_cols, cat_cols

def _logit_pipeline(num_cols: list,
                    cat_cols: list,
                    logit_imputer_strategy: str = "median"):
    """
    [pt] Constrói o pipeline de pré-processamento e classificação para o modelo de Regressão Logística.
         O pipeline aplica imputação e escalonamento nas features numéricas, e imputação e One-Hot Encoding
         nas features categóricas.
    [en] Builds the preprocessing and classification pipeline for the Logistic Regression model.
         The pipeline applies imputation and scaling to numerical features, and imputation and One-Hot Encoding
         to categorical features.

    Args:
    num_cols: list
        [pt] Lista de colunas numéricas.
        [en] List of numerical feature names.
    cat_cols: list
        [pt] Lista de colunas categóricas.
        [en] List of categorical feature names.
    logit_imputer_strategy: str
        [pt] Estratégia de imputação para colunas numéricas. O padrão é "median".
        [en] Imputation strategy for numerical features. Default is "median".

    Returns:
        [pt] Pipeline do scikit-learn contendo o pré-processador (ColumnTransformer) e o classificador (LogisticRegression).
        [en] scikit-learn Pipeline containing the preprocessor (ColumnTransformer) and the classifier (LogisticRegression).

    Raises:
        [pt] ValueError: Se ambas as listas num_cols e cat_cols estiverem vazias.
        [en] ValueError: If both num_cols and cat_cols are empty.
    """

    transformers = []

    # [pt] Adiciona o transformer numérico se houver colunas numéricas
    # [en] Adds the numerical transformer if there are numerical columns
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=logit_imputer_strategy)),
            ("scaler", StandardScaler()) # [pt] Escalonamento para Regressão Logística traz vantagem computacional, mas perde interpretabilidade dos coeficientes
                                         # [en] Scaling in Logistic Regression improves computational efficiency, but reduces coefficient interpretability
        ])
        transformers.append(("num", num_pipe, num_cols))

    # [pt] Adiciona o transformer categórico se houver colunas categóricas
    # [en] Adds the categorical transformer if there are categorical columns
    if cat_cols:
        cat_pipe = Pipeline([
            # [pt] Substitui valores nulos pela categoria "missing" antes do OHE
            # [en] Replaces null values with a "missing" category before OHE
            ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
            ("encoder", OneHotEncoder(sparse_output=False)) # [pt] 'sparse_output=False' retorna um array NumPy em vez de uma matriz esparsa, garantindo compatibilidade com pandas
                                                            # [en] 'sparse_output=False' returns a NumPy array instead of a sparse matrix, ensuring compatibility with pandas
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    if not transformers:
        # [pt] Caso a lista esteja vazia, interrompe o pipeline para evitar treinamento com DataFrame vazio.
        # [en] If the transformer list is empty, stops the pipeline to prevent fitting on an empty DataFrame.
        raise ValueError(
            f"[pt] ERRO CRÍTICO: O DataFrame não possui colunas numéricas ou categóricas.\n"
            f"[en] CRITICAL ERROR: DataFrame has no numerical or categorical columns."
            )

    # [pt] 'remainder=drop' exclui do pipeline as colunas não tratadas pelo pré-processador
    # [en] 'remainder=drop' excludes from the pipeline any columns not handled by the preprocessor
    preprocessor = ColumnTransformer(transformers=transformers,
                             remainder="drop")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000,
                                          class_weight=None,
                                          random_state=42,
                                          penalty=None))
    ])

def _rf_pipeline(num_cols: list,
                 cat_cols: list):
    """
    [pt] Constrói o pipeline de pré-processamento e classificação para o modelo de Random Forest.
         O pipeline aplica a imputação e One-Hot Encoding nas features categóricas.
    [en] Builds the preprocessing and classification pipeline for the Random Forest model.
         The pipeline applies imputation and One-Hot Encoding to categorical features.

    Args:
    num_cols: list
        [pt] Lista de colunas numéricas.
        [en] List of numerical feature names.
    cat_cols: list
        [pt] Lista de colunas categóricas.
        [en] List of categorical feature names.

    Returns:
        [pt] Pipeline do scikit-learn contendo o pré-processador (ColumnTransformer) e o classificador (RandomForestClassifier).
        [en] scikit-learn Pipeline containing the preprocessor (ColumnTransformer) and the classifier (RandomForestClassifier).

    Raises:
        [pt] ValueError: Se ambas as listas num_cols e cat_cols estiverem vazias.
        [en] ValueError: If both num_cols and cat_cols are empty.
    """

    transformers = []

    if num_cols:
        # [pt] RF >= sklearn 1.4 trata NaN nativamente, por isso não há transformação
        # [en] RF >= sklearn 1.4 handles NaN natively, so no imputation step is needed
        transformers.append(("num", "passthrough", num_cols))

    # [pt] Adiciona o transformer categórico se houver colunas categóricas
    # [en] Adds the categorical transformer if there are categorical columns
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    if not transformers:
        # [pt] Caso a lista esteja vazia, interrompe o pipeline para evitar treinamento com DataFrame vazio.
        # [en] If the transformer list is empty, stops the pipeline to prevent fitting on an empty DataFrame.
        raise ValueError(
            f"[pt] ERRO CRÍTICO: O DataFrame não possui colunas numéricas ou categóricas.\n"
            f"[en] CRITICAL ERROR: DataFrame has no numerical or categorical columns."
            )

    # [pt] 'remainder=drop' exclui do pipeline as colunas não tratadas pelo pré-processador
    # [en] 'remainder=drop' excludes from the pipeline any columns not handled by the preprocessor
    preprocessor = ColumnTransformer(transformers,
                             remainder="drop")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])


def _train_logit(X_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_train: pd.Series,
                 num_cols: list,
                 cat_cols: list):
    """
    [pt] Treina o modelo de Regressão Logística e retorna o modelo treinado e as probabilidades preditas.
    [en] Trains the Logistic Regression model and returns the fitted model and predicted probabilities.

    Args:
    X_train: pd.DataFrame
        [pt] Features do conjunto de treino.
        [en] Training set features.
    X_test: pd.DataFrame
        [pt] Features do conjunto de teste.
        [en] Test set features.
    y_train: pd.Series
        [pt] Target do conjunto de treino.
        [en] Training set target.
    num_cols: list
        [pt] Lista de features numéricas.
        [en] List of numerical feature names.
    cat_cols: list
        [pt] Lista de features categóricas.
        [en] List of categorical feature names.

    Returns:
        [pt] Dicionário contendo o modelo treinado e as probabilidades preditas para treino e teste.
        [en] Dictionary containing the fitted model and predicted probabilities for train and test sets.
    """

    pipe = _logit_pipeline(num_cols, cat_cols)

    pipe.fit(X_train, y_train)

    return {
        "model":       pipe,
        "proba_train": pipe.predict_proba(X_train)[:, 1],
        "proba_test":  pipe.predict_proba(X_test)[:, 1],
    }

def _train_rf(X_train: pd.DataFrame,
              X_test: pd.DataFrame,
              y_train: pd.Series,
              num_cols: list,
              cat_cols: list,
              param_dist: dict = default_rf_params,
              rf_n_iter: int = 20,
              rf_cv: int = 4,
              random_state: int = 42,
              rf_verbose: int = 1):
    """
    [pt] Treina o modelo de Random Forest com busca de hiperparâmetros via RandomizedSearchCV
         e retorna o melhor modelo encontrado e as probabilidades preditas.
    [en] Trains the Random Forest model with hyperparameter search via RandomizedSearchCV
         and returns the best fitted model and predicted probabilities.

    Args:
    X_train: pd.DataFrame
        [pt] Features do conjunto de treino.
        [en] Training set features.
    X_test: pd.DataFrame
        [pt] Features do conjunto de teste.
        [en] Test set features.
    y_train: pd.Series
        [pt] Target do conjunto de treino.
        [en] Training set target.
    num_cols: list
        [pt] Lista de features numéricas.
        [en] List of numerical feature names.
    cat_cols: list
        [pt] Lista de features categóricas.
        [en] List of categorical feature names.
    param_dist: dict
        [pt] Dicionário com as distribuições de hiperparâmetros para a busca aleatória. O padrão é default_rf_params.
        [en] Dictionary of hyperparameter distributions for random search. Default is default_rf_params.
    rf_n_iter: int
        [pt] Número de combinações de hiperparâmetros a serem avaliadas. O padrão é 20.
        [en] Number of hyperparameter combinations to evaluate. Default is 20.
    rf_cv: int
        [pt] Número de folds para a validação cruzada. O padrão é 4.
        [en] Number of cross-validation folds. Default is 4.
    random_state: int
        [pt] Semente aleatória para reprodutibilidade. O padrão é 42.
        [en] Random seed for reproducibility. Default is 42.
    rf_verbose: int
        [pt] Nível de verbosidade durante a busca. O padrão é 1.
        [en] Verbosity level during the search. Default is 1.

    Returns:
        [pt] Dicionário contendo o melhor modelo, as probabilidades preditas para treino e teste,
             os melhores hiperparâmetros encontrados e o AUC médio do CV.
        [en] Dictionary containing the best fitted model, predicted probabilities for train and test sets,
             the best hyperparameters found, and the mean CV AUC score.
    """

    pipe = _rf_pipeline(num_cols, cat_cols)

    # [pt] Prefixar com "classifier__" para busca dentro do Pipeline
    # [en] Prefix with "classifier__" to search within the Pipeline
    param_dist_prefixed = {f"classifier__{k}": v for k, v in param_dist.items()}

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist_prefixed,
        n_iter=rf_n_iter,
        cv=rf_cv,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
        verbose=rf_verbose,
    )

    search.fit(X_train, y_train)

    best = search.best_estimator_

    return {
        "model":       best,
        "proba_train": best.predict_proba(X_train)[:, 1],
        "proba_test":  best.predict_proba(X_test)[:, 1],
        "best_params": search.best_params_,
        "best_score": search.best_score_ # [pt] AUC médio do CV — comparar com AUC do teste para verificar overfitting
                                         # [en] Mean CV AUC — compare with test AUC to check for overfitting
    }

def train_models(df: pd.DataFrame,
                 target_col: str,
                 test_size: float = 0.20,
                 random_state: int = 42,
                 logistic_regression: bool = True,
                 logit_imputer_strategy: str = "median",
                 random_forest: bool = True,
                 rf_param_dist: dict = default_rf_params,
                 rf_n_iter: int = 20,
                 rf_cv: int = 4,
                 rf_verbose: int = 1):
    """
    [pt] Orquestra o treinamento dos modelos de Regressão Logística e Random Forest,
         executando o pré-processamento, a divisão dos dados e o treinamento de cada modelo selecionado.
    [en] Orchestrates the training of Logistic Regression and Random Forest models,
         handling preprocessing, data splitting, and fitting of each selected model.

    Args:
    df: pd.DataFrame
        [pt] DataFrame de entrada contendo features e target.
        [en] Input DataFrame containing features and target.
    target_col: str
        [pt] Nome da coluna target.
        [en] Name of the target column.
    test_size: float
        [pt] Proporção dos dados a ser utilizada como conjunto de teste. O padrão é 0.2.
        [en] Fraction of the data to use as the test set. Default is 0.2.
    random_state: int
        [pt] Semente aleatória para reprodutibilidade. O padrão é 42.
        [en] Random seed for reproducibility. Default is 42.
    logistic_regression: bool
        [pt] Se True, treina o modelo de Regressão Logística. O padrão é True.
        [en] If True, trains the Logistic Regression model. Default is True.
    logit_imputer_strategy: str
        [pt] Estratégia de imputação para features numéricas no modelo de Regressão Logística. O padrão é "median".
        [en] Imputation strategy for numerical features in the Logistic Regression pipeline. Default is "median".
    random_forest: bool
        [pt] Se True, treina o modelo de Random Forest. O padrão é True.
        [en] If True, trains the Random Forest model. Default is True.
    rf_param_dist: dict
        [pt] Dicionário com as distribuições de hiperparâmetros para a busca aleatória do Random Forest. O padrão é default_rf_params.
        [en] Dictionary of hyperparameter distributions for Random Forest random search. Default is default_rf_params.
    rf_n_iter: int
        [pt] Número de combinações de hiperparâmetros a serem avaliadas. O padrão é 20.
        [en] Number of hyperparameter combinations to evaluate. Default is 20.
    rf_cv: int
        [pt] Número de folds para a validação cruzada do Random Forest. O padrão é 4.
        [en] Number of cross-validation folds for the Random Forest search. Default is 4.
    rf_verbose: int
        [pt] Nível de verbosidade durante a busca de hiperparâmetros. O padrão é 1.
        [en] Verbosity level during hyperparameter search. Default is 1.

    Returns:
        [pt] Dicionário contendo os resultados de cada modelo treinado e os conjuntos y_train e y_test.
        [en] Dictionary containing the results of each fitted model and the y_train and y_test splits.
    """

    X_train, X_test, y_train, y_test = _split_data(
        df,
        target_col,
        test_size=test_size,
        random_state=random_state
    )

    num_cols, cat_cols = _get_features_lists(X_train)

    return {
        "Logistic Regression": _train_logit(X_train, X_test, y_train, num_cols, cat_cols),

        "Random Forest":    _train_rf(X_train, X_test, y_train, num_cols, cat_cols,
                           rf_param_dist, rf_n_iter, rf_cv, random_state, rf_verbose),

        "y_train": y_train,
        "y_test":  y_test,
    }
