Contenido de la actividad

Deberán participar en la siguiente competencia de Kaggle:

Rossmann Store Sales



Tendrán que mejorar en el private score de la competencia el valor: 0.129



Recomendamos clonar el siguiente repositorio:

https://github.com/deeplearning-itba/kaggle-rossmann



El dataset está en el archivo rossmann.zip (Recomendamos descombrimirlo en la carpeta dataset. Caso contrario cambiar el path en las notebooks



Corriendo las primeras 4 notebooks se grabarán los archivos: 

train_normalized_data.fth y test_normalized_data.fth



Estos archivos tiene los datos de train y test normalizados y pre-preprocesados para entrenar cualquiera de los modelos. Tiene total libertad de modificar las primeras 4 notebooks y hacer algún tipo de pre-procesamiento diferente. (Una recomendación puede ser no borrar las ventas con valor 0 -store cerrado-)



Estos archivos serán cargados por las notebooks 6, 7, 8 (Red Neuronal, XGBoost, LightGBM) donde hay armando una ejemplo de cada uno de los modelos. Pruebe los distintos modelos con las distintas opciones de hiperparámetros





Nota: Para entrenar XGBoost con la opción de embeddings tiene que correr la notebook 6 hasta el final



Deberán hacer un submit a la competencia de Kaggle y una vez mejorado el score nos deben enviar un screenshot al correo: deeplearning@itba.edu.ar

