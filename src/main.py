from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.lithology_analyzer import LithologyAnalyzer
from src.visualization import Visualizer
from src.models import LithologyPredictor

def main():
    # Загрузка данных
    print("Загрузка данных...")
    loader = DataLoader()
    data = loader.load_well_logs()
    lithology_data = loader.load_lithology_data()
    
    # Предобработка данных
    print("Предобработка данных...")
    preprocessor = DataPreprocessor()
    well_logs, thermal_logs = preprocessor.prepare_well_logs(data)
    lithology = preprocessor.prepare_lithology_data(lithology_data)
    
    # Объединение данных каротажа
    well_logs_merged = preprocessor.merge_well_logs(well_logs)
    
    # Анализ литологии
    print("Анализ литологии...")
    analyzer = LithologyAnalyzer()
    final_data = analyzer.merge_with_lithology(lithology, well_logs_merged)
    
    # Визуализация
    print("Визуализация данных...")
    visualizer = Visualizer()
    visualizer.plot_lithology_distributions(final_data)
    visualizer.plot_correlation_heatmap(final_data)
    
    # Моделирование (опционально)
    print("Обучение моделей...")
    predictor = LithologyPredictor()
    X = final_data[['ГК ', 'ГГпК', 'ПС', 'КС']]
    y = final_data['Литология']
    
    results = predictor.train_models(X, y)
    
    print("\nАнализ завершен!")
    return final_data, results

if __name__ == "__main__":
    final_data, results = main()
