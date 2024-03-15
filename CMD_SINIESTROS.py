# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:29:18 2024

@author: UTM
"""

#IMPORTAR FUNCIONES
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
import category_encoders as ce
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree


def datosSiniestro():
    url_df='https://raw.githubusercontent.com/Emax1/UNTUMBES/main/Transporte_2021.csv'
    df=pd.read_csv(url_df,sep=';')
    df.info()
    # Verificar datos perdidos
    datos_perdidos = df.isnull().sum()
    #df = df.drop(columns='ANIO')
    # Imprimir los datos perdidos por columna
    print("Datos perdidos por columna:")
    print(datos_perdidos)
    return df

def grafico_barra(df,Variable):
    # Contar la frecuencia de cada categoría
    frecuencia_categorias = df[Variable].value_counts()
    
    # Crear el gráfico de barras
    plt.figure(figsize=(8, 6))
    frecuencia_categorias.plot(kind='bar', color='skyblue')
    plt.title('Distribución de Categorías')
    plt.xlabel('Categoría')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
def grafico_barra(df,Variable):
    # Contar la frecuencia de cada categoría
    frecuencia_categorias = df[Variable].value_counts()
    
    # Crear el gráfico de barras
    plt.figure(figsize=(8, 6))
    frecuencia_categorias.plot(kind='bar', color='skyblue')
    plt.title('Distribución de Categorías')
    plt.xlabel('Categoría')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
def BinaryEncoderVar(df,columnas):
    # Supongamos que tienes un DataFrame llamado 'df' con una columna 'nivel'
    # Creamos una instancia de BinaryEncoder y la ajustamos a los datos
    encoder = ce.BinaryEncoder(cols=columnas)
    df_binary_encoded = encoder.fit_transform(df[columnas])
    
    df_binary_encoded.info()
    
    # Concatenamos el DataFrame codificado con el DataFrame original
    df_encoded = pd.concat([df, df_binary_encoded], axis=1)
    df_encoded.info()
    # Lista de nombres de columnas a eliminar
    #columnas_a_eliminar = columnas
    
    # Eliminar las columnas
    df_encoded = df_encoded.drop(columns=columnas)
    df_encoded.info()
    return df_encoded

def patrones(df_encoded,k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df_encoded)
    labels = kmeans.predict(df_encoded)
    
    ###Viviendas por clase
    Counter(labels)
    
    #presentación de las clases
    # Ajusta el modelo PCA
    pca = PCA()
    pca.fit(df_encoded)
    
    # Obtén los loadings
    #loadings = pca.components_
    
    componentes_principales = pca.transform(df_encoded)
    
    
    
    # Creamos el scatter plot
    #ax.scatter(componentes_principales[:,0], componentes_principales[:,1], componentes_principales[:,2], c=labels, label='Componentes principales')
    plt.scatter(componentes_principales[:,0], componentes_principales[:,1], c=labels)
    
    # Agregamos etiquetas y título
    plt.xlabel('Variable X')
    plt.ylabel('Variable Y')
    plt.title('Gráfico de Puntos con Color por Clase')
    
    
    ################
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(componentes_principales[:,0], componentes_principales[:,1], componentes_principales[:,2], c=labels, label='Componentes principales')
    # Ajustamos la posición del gráfico
    ax.set_position([0.1, 0.1, 0.6, 0.6])  # [left, bottom, width, height]
    #ax.set_xlim([-3, 3])
    #ax.set_zlim([-3, 3])
    #ax.set_ylim([-3, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.title('PCA en tres dimensiones')
    plt.legend()
    return labels

def caracteristica_cluster(X,y):
    # Inicializar el clasificador de árbol de decisión
    clf = DecisionTreeClassifier()
    
    # Entrenar el clasificador con los datos completos
    clf.fit(X, y)
    
    # Visualizar el árbol de decisión
    plt.figure(figsize=(15, 10))
    plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names="clase")
    # Guardar la imagen en el directorio actual con el nombre 'grafico.png'
    plt.savefig('grafico.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    
    ######DEFINIR PESOS POR RANDOM FOREST
    # Cargar el conjunto de datos de Iris
    #X = df_encoded
    #y = etiquetas
    
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear un modelo de Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Entrenar el modelo
    rf_model.fit(X_train, y_train)
    
    # Obtener la importancia de las características
    importances = rf_model.feature_importances_
    
    # Crear un DataFrame para mostrar la importancia de las características
    feature_importance_df = pd.DataFrame(importances, index=X.columns, columns=['Importance'])
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Mostrar la importancia de las características
    print("Importancia de las características:")
    print(feature_importance_df)
    
    # Graficar la importancia de las características
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df.index, feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.show()
