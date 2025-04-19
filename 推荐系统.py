import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data = pd.read_csv(r"E:\大三上\点宽杯\111.csv")
# 1. 数据预处理
def data_preprocessing(data):
    # 假设数据已经清洗和特征工程完成
    return data

# 2. 用户画像构建和商品分析
def build_profiles_and_analysis(data):
     data['年龄段'] = pd.cut(data['年龄'], bins=[0, 18, 24, 34, 60, 100], labels=['<18', '18-24', '25-34', '35-59', '60+'])
     data['性别'] = data['性别'].astype('category')
     user_behavior_counts = data.groupby('用户ID')['行为'].value_counts()
     # 用户消费能力分析
     scaler = StandardScaler()
     data['经济权重标准化'] = scaler.fit_transform(data[['经济权重']])
     kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)
     kmeans.fit(data[['经济权重标准化']])
     region_mapping = {
         '北京': '华北', '天津': '华北', '河北': '华北', '山西': '华北', '内蒙古': '华北',
         '上海': '华东', '江苏': '华东', '浙江': '华东', '安徽': '华东', '福建': '华东', '江西': '华东', '山东': '华东',
         '广东': '华南', '广西': '华南', '海南': '华南', '香港': '华南', '澳门': '华南',
         '河南': '华中', '湖北': '华中', '湖南': '华中',
         '重庆': '西南', '四川': '西南', '贵州': '西南', '云南': '西南', '西藏': '西南',
         '陕西': '西北', '甘肃': '西北', '青海': '西北', '宁夏': '西北', '新疆': '西北',
         '台湾': '华东','黑龙江': '东北', '吉林': '东北', '辽宁': '东北'}
     data['区域'] = data['省份'].map(region_mapping)
     data['消费能力等级'] = kmeans.labels_
     # 构建用户画像
     user_profile = data.groupby('用户ID').agg({
         '年龄段': 'first',
         '性别': 'first',
         '区域': 'first',
         '商品类型': lambda x: ' '.join(x.unique()),  
         '具体商品': lambda x: ' '.join(x.unique())   
     }).reset_index()
#return user_profile
# 3. 相似度计算
def calculate_similarity(data):
    # 计算用户之间的相似度
    user_item_matrix = data.pivot_table(index='用户ID', columns='商品ID', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    return user_similarity

    # 计算商品之间的相似度
    product_similarity = cosine_similarity(data[['feature1', 'feature2', 'feature3']])
    return product_similarity

# 4. 推荐算法选择
def collaborative_filtering(data, user_similarity, product_similarity):
    # 使用协同过滤算法进行推荐
    pass

def content_based_filtering(data, product_analysis):
    # 使用基于内容的推荐算法进行推荐
    pass

# 5. 模型训练与测试
def train_test_model(data):
    X_train, X_test, y_train, y_test = train_test_split(data['features'], data['labels'], test_size=0.2, random_state=42)
    model = NearestNeighbors()
    model.fit(X_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# 6. 结果输出
def generate_recommendations(user_similarity, product_similarity, user_profiles, product_analysis):
    # 根据模型为用户生成推荐列表
    pass

# 主流程
if __name__ == "__main__":
    # 预处理数据
    data = pd.read_csv("E:/大三上/点宽杯/111.csv")
    preprocessed_data = data_preprocessing(data)
    
    # 构建用户画像和商品分析
    user_profiles, product_analysis = build_profiles_and_analysis(preprocessed_data)
    
    # 计算相似度
    user_similarity = calculate_similarity(preprocessed_data)
    product_similarity = calculate_similarity(preprocessed_data[['feature1', 'feature2', 'feature3']])
    
    # 训练模型并测试
    accuracy = train_test_model(preprocessed_data)
    print(f"Model accuracy: {accuracy}")
    
    # 生成推荐
    recommendations = generate_recommendations(user_similarity, product_similarity, user_profiles, product_analysis)
    print(recommendations)
