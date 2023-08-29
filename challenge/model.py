import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import plot_importance
from datetime import datetime
from typing import Tuple, Union, List, Dict

class DelayModel:

    def __init__(self):
        self._model = None

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        # Feature Engineering and Preprocessing

        #print(data, flush=True)
        
        def get_period_day(date):
            date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
            morning_min = datetime.strptime("05:00", '%H:%M').time()
            morning_max = datetime.strptime("11:59", '%H:%M').time()
            afternoon_min = datetime.strptime("12:00", '%H:%M').time()
            afternoon_max = datetime.strptime("18:59", '%H:%M').time()
            evening_min = datetime.strptime("19:00", '%H:%M').time()
            evening_max = datetime.strptime("23:59", '%H:%M').time()
            night_min = datetime.strptime("00:00", '%H:%M').time()
            night_max = datetime.strptime("4:59", '%H:%M').time()

            if(date_time > morning_min and date_time < morning_max):
                return 'mañana'
            elif(date_time > afternoon_min and date_time < afternoon_max):
                return 'tarde'
            elif(
                (date_time > evening_min and date_time < evening_max) or
                (date_time > night_min and date_time < night_max)
            ):
                return 'noche'

        data['period_day'] = data['Fecha-I'].apply(get_period_day)

        def is_high_season(fecha):
            fecha_año = int(fecha.split('-')[0])
            fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
            range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
            range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
            range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
            range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
            range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
            range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
            range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
            range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
            
            if ((fecha >= range1_min and fecha <= range1_max) or 
                (fecha >= range2_min and fecha <= range2_max) or 
                (fecha >= range3_min and fecha <= range3_max) or
                (fecha >= range4_min and fecha <= range4_max)):
                return 1
            else:
                return 0

        data['high_season'] = data['Fecha-I'].apply(is_high_season)

        def get_min_diff(data):
            fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
            fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
            min_diff = ((fecha_o - fecha_i).total_seconds())/60
            return min_diff


        data['min_diff'] = data.apply(get_min_diff, axis=1)

        threshold_in_minutes = 15
    
        if target_column:
            data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
            return data.drop(columns=[target_column])
        else:
            return data


    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        # Model Training
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

        # Initialize and fit the model (XGBoost)
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        self._model.fit(x_train, y_train)

        # Print classification report
        y_pred = self._model.predict(x_test)
        print(classification_report(y_test, y_pred))


    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        # Predict using the trained model
        predictions = self._model.predict(features)

        return predictions



if __name__ == "__main__":
    # Crear una instancia de la clase DelayModel
    model_instance = DelayModel()

    # Cargar datos del DataFrame 
    data_path = '/Users/gonzaloordenes/Documents/Projects/challenge/data/data.csv'
    #data = pd.read_csv(data_path)

    data = pd.read_csv(data_path, dtype={'Vlo-O': str, 'Vlo-I': str})

    data['Vlo-O'] = pd.to_numeric(data['Vlo-O'], errors='coerce').fillna(0).astype(int)

    
    #print(data['Vlo-I'].unique())

    # Llamar a la función preprocess en la instancia de la clase
    preprocessed_data = model_instance.preprocess(data)

    # Imprimir los primeros registros de los datos preprocesados
    print(preprocessed_data.head())
