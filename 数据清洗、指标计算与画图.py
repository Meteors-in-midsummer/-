import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import chi2_contingency	 
customers = pd.read_csv(r"E:\大三上\点宽杯\用户信息(id所在地年龄性别权益）.csv")
user_behavior = pd.read_csv(r"E:\大三上\点宽杯\用户行为商品时间备注.csv")
products = pd.read_csv(r"E:\大三上\点宽杯\商品（id类型具体商品成本单价库存）.csv")
promotions = pd.read_csv(r"E:\大三上\点宽杯\日期周末假期折扣量.csv")
locations = pd.read_csv(r"E:\大三上\点宽杯\省份城市经济权重.csv")
#对locaions进行数据清洗
print("数据概况：")
print(locations.info())  
print("\n数据的前几行：")
print(locations.head())
print("\n缺失值统计：")
print(locations.isnull().sum())
locations['经济权重'] = locations['经济权重'].fillna(locations['经济权重'].median())  
locations.dropna(subset=['省份', '城市'], inplace=True)
print("\n重复值统计：")
print(locations.duplicated().sum())
locations = locations.drop_duplicates()
locations['省份'] = locations['省份'].str.strip().str.title()  
locations['城市'] = locations['城市'].str.strip().str.title()  
print("\n经济权重描述：")
print(locations['经济权重'].describe())
locations = locations[locations['经济权重'] >= 0]  # 去除负值
locations = locations[locations['经济权重'] <= locations['经济权重'].quantile(0.95)]  
cleaned_file_path = r"E:\大三上\点宽杯\省份城市经济权重111.csv"
locations.to_csv(cleaned_file_path, index=False)
print(f"\n数据清洗完成，已保存为：{cleaned_file_path}")#清洗之后的文件保存为111
user_behavior['时间'] = pd.to_datetime(user_behavior['时间'])
customers.fillna({'年龄': 0, '性别': '未知', '权益': '非VIP'}, inplace=True)
min_age = 0
max_age = 120
outliers = (customers['年龄'] < min_age) | (customers['年龄'] > max_age)
# 用中位数年龄替换异常值
median_age = customers['年龄'].median()
customers.loc[outliers, '年龄'] = median_age
customers.to_csv(r"E:\大三上\点宽杯\用户信息异常值清洗.csv", index=False)
products.fillna({'成本': 0, '单价': 0, '库存': 0}, inplace=True)
promotions.fillna({'折扣量': 0}, inplace=True)
user_behavior_products = pd.merge(user_behavior, products, on='商品ID', how='left')#解释了用户购买了什么商品类型下的什么东西
#销售规律分析
# 按商品类型和日期聚合销售数据
# 相关性分析
sales_data = user_behavior_products[user_behavior_products['行为'] == '购买'].groupby(['商品类型', pd.Grouper(key='时间', freq='M')])['单价'].sum().reset_index()
sales_data.rename(columns={'单价': '月销售额'}, inplace=True)#给出每种商品类型的月销售总额
sales_data_corr = sales_data.groupby('商品类型')['月销售额'].corr(promotions.groupby('日期')['折扣量'].mean().reset_index(drop=True))
print("各类商品月销售额与平均折扣量的相关性：")
print(sales_data_corr) 
daily_active_users = user_behavior['时间'].value_counts().sort_index()#活跃度（DAU，Daily Active Users）：每日活跃用户数量
daily_active_users_df = pd.DataFrame(list(daily_active_users.items()), columns=['时间', '活跃度'])# 将结果转换为DataFrame以便查看
daily_active_users = user_behavior.groupby(user_behavior['时间'].dt.date)['用户ID'].nunique()
user_behavior['时间'] = pd.to_datetime(user_behavior['时间'])
user_behavior['first_action_date'] = user_behavior.groupby('用户ID')['时间'].transform('min')
new_users = user_behavior[user_behavior['时间'] == user_behavior['first_action_date']]
user_behavior['retained_7_days'] = user_behavior.apply(
    lambda row: 1 if row['用户ID'] in new_users['用户ID'].values and \
(row['时间'] - row['first_action_date']).days == 7 else 0,axis=1)
user_retention = user_behavior.groupby('用户ID')['retained_7_days'].max().reset_index()
retention_rate = user_retention['retained_7_days'].mean()
bounce_users_daily = user_behavior[user_behavior['行为'] == '浏览'].groupby('时间')['用户ID'].apply(lambda x: x.unique())
total_users_daily = user_behavior.groupby('时间')['用户ID'].nunique()# 计算每天的总用户数
bounce_rate_daily = {}# 计算每日跳失率
for date, users in bounce_users_daily.items():
    total_users = total_users_daily.loc[date]
    bounce_rate = len(users) / total_users if total_users > 0 else 0
    bounce_rate_daily[date] = bounce_rate
bounce_rate_daily_df = pd.DataFrame(list(bounce_rate_daily.items()), columns=['时间', '跳失率'])# 将结果转换为DataFrame以便查看
purchasing_users = user_behavior[user_behavior['行为'] == '购买']['用户ID'].unique()#用户转化率
conversion_rate = len(purchasing_users) / len(user_behavior['用户ID'].unique()) if len(user_behavior['用户ID'].unique()) > 0 else 0
user_growth_rate = []#用户增长率
all_new_users = user_behavior.groupby(user_behavior['时间'].dt.date)['用户ID'].nunique()  
for i in range(1, len(all_new_users)):
    new_users_day0 = all_new_users.iloc[i - 1]
    new_users_day1 = all_new_users.iloc[i]
    growth_rate = (new_users_day1 - new_users_day0) / new_users_day0 if new_users_day0 > 0 else 0
    user_growth_rate.append(growth_rate)
last_active_date = user_behavior.groupby('用户ID')['时间'].max().reset_index()#客户流失率
inactive_users = last_active_date[last_active_date['时间'].dt.date < (max(user_behavior['时间'].dt.date) - timedelta(days=30))]['用户ID'].unique()
churn_rate = len(inactive_users) / len(user_behavior['用户ID'].unique()) if len(user_behavior['用户ID'].unique()) > 0 else 0
#展示用户行为指标
print("\n用户行为指标：")
print("活跃度（DAU）：", daily_active_users)  # 仅展示最后一天的DAU作为示例
print("7日留存率：", retention_rate)
print("用户转化率：", conversion_rate)
print("用户增长率：", np.mean(user_growth_rate))  
print("客户流失率：", churn_rate)
user_behavior_purchase = user_behavior[user_behavior['行为'] == '购买']
user_behavior_purchase = pd.merge(user_behavior_purchase, products[['商品ID', '商品类型']], on='商品ID', how='left')
user_behavior_purchase_category = user_behavior_purchase.groupby(['用户ID', '商品类型'])['商品ID'].count().reset_index()
user_behavior_purchase_category.rename(columns={'商品ID': '购买次数'}, inplace=True)
user_most_bought_category = user_behavior_purchase_category.loc[user_behavior_purchase_category.groupby('用户ID')['购买次数'].idxmax()]
user_most_bought_category = user_most_bought_category[['用户ID', '商品类型']]
print("\n每个用户最常购买的商品类别：")
print(user_most_bought_category.head())
#RFM建模
#R的时间间隔计算
user_behavior_products = pd.merge(user_behavior, products, on='商品ID', how='left')
user_behavior_products['时间'] = pd.to_datetime(user_behavior_products['时间'])
user_behavior_products_purchases = user_behavior_products[user_behavior_products['行为'] == '购买']
user_behavior_products_recent_purchases = user_behavior_products_purchases.groupby('用户ID')['时间'].max().reset_index()
target_date = pd.to_datetime('2024-11-15')
user_behavior_products_recent_purchases['R'] = (target_date - user_behavior_products_recent_purchases['时间']).dt.days
R=user_behavior_products_recent_purchases[['用户ID', 'R']]
#F的下单次数计算
order_counts = user_behavior_products_purchases.groupby('用户ID').size()
F = order_counts.reset_index(name='F')
#M的平均消费金额计算
average_spending = user_behavior_products_purchases.groupby('用户ID')['单价'].mean()
M = average_spending.reset_index(name='M')
#合并RFM序列
RFM=R[["用户ID","R"]].merge(F,on="用户ID").merge(M,on="用户ID")
pd.set_option('display.unicode.east_asian_width',True)
# 重设index
RFM.set_index("用户ID",inplace=True)
# 将RFM模型的原始值处理成对应的等级
def RFM_R(value):
    if value <= 50:
        return 5
    elif value <= 100:
        return 4
    elif value <= 300:
        return 3
    elif value <= 500:
        return 2
    return 1
 
def RFM_F(value):
     if value <= 5:
        return 1
     elif value <= 10:
        return 2
     elif value <= 20:
        return 3
     elif value <= 35:
        return 4
     return 5
   
def RFM_M(value):
    if value < 300:
        return 1
    elif value < 500:
        return 2
    elif value < 700:
        return 3
    elif value < 900:
        return 4
    return 5
RFM['R'] = RFM.R.apply(RFM_R)
RFM['F'] = RFM.F.apply(RFM_F)
RFM['M'] = RFM.M.apply(RFM_M)
column_means = RFM.mean()
RFM_model = RFM.apply(lambda x: x.apply(lambda xi: '高' if xi > column_means[x.name] else '低'))
RFM_model['Tag'] = RFM_model.sum(axis=1)
RFM_model['Tag'] = RFM_model.Tag.astype('category').cat.reorder_categories(['高高高', '高低高', '低高高', '低低高', '高高低', '高低低', '低高低', '低低低'])
RFM_model['Tag'] = RFM_model['Tag'].cat.rename_categories({'高高高': '高价值客户','高低高':'重点发展客户','低高高':'重点保持客户', '低低高':'重点挽留客户', '高高低':'一般价值客户', '高低低':'一般发展客户', '低高低':'一般保持客户', '低低低':'潜在客户'})
result = RFM_model.Tag.value_counts().sort_index()
font = {"family": "Microsoft Yahei"}
plt.rc("font",**font)
result.plot(kind='bar')
#plt.show()
plt.rcParams['font.sans-serif'] = ['SimHei']
pd.set_option('display.unicode.east_asian_width',True)
users = pd.read_csv("E:\\大三上\\点宽杯\\用户信息异常值清洗.csv")
locations = pd.read_csv("E:\\大三上\\点宽杯\\省份城市经济权重.csv")
locations.rename(columns={'城市': '用户所在地'}, inplace=True)
df_merged = pd.merge(users,locations, on='用户所在地', how='left')
user_behavior_products=pd.read_csv(r"E:\大三上\点宽杯\用户行为买了什么商品.csv")
df=pd.merge(df_merged,user_behavior_products,on='用户ID',how='left')
sales_area=df.groupby('省份')['单价'].sum()
# 各地区销售额
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(sales_area, labels=sales_area.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
for text in texts:
    text.set_rotation(30)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_size(6)
plt.title('各地区销售额')
# 不同商品类型销售额占比
sales_type=df.groupby('商品类型')['单价'].sum()
plt.figure(figsize=(8, 8))  
plt.pie(sales_type, labels=sales_type.index, autopct='%1.1f%%', startangle=140)
plt.title('不同商品类型销售额占比')  
plt.axis('equal')  
#总订单占比
member_count = df[df['权益'] == '会员'].shape[0] 
non_member_count = df[df['权益'] != '会员'].shape[0]
total_count = member_count + non_member_count
member_ratio = member_count / total_count
non_member_ratio = non_member_count / total_count
labels = ['会员', '非会员']
sizes = [member_ratio, non_member_ratio]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.set_title('总订单占比')
#总消费金额占比
grouped_df = df.groupby('权益')['单价'].sum()
fig, ax = plt.subplots()
ax.pie(grouped_df, labels=grouped_df.index, autopct='%1.1f%%', startangle=90)
ax.set_title('总消费金额占比')
#会员的年龄分布
members = users[users['权益'] == '会员']
def age_group(age):
    if age < 18:
        return '未成年人'
    elif 18 <= age <= 44:
        return '年轻人'
    elif 45 <= age <= 60:
        return '中年人'
    else:
        return '老年人'
members_copy = members.copy()
members_copy['年龄组'] = members_copy['年龄'].apply(age_group)
age_group_counts = members_copy['年龄组'].value_counts()
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.90, wedgeprops=dict(width=0.3))
for text in texts:
    text.set_rotation(35)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_size(8)
ax.set_title('会员的年龄分布')
#会员的男女比例
members.loc[:, '性别'] = pd.Categorical(members['性别'], categories=['男', '女'], ordered=True)
gender_counts = members['性别'].value_counts()
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
for text in texts:
    text.set_rotation(35)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_size(8)
ax.set_title('会员的男女比例')
#男女消费金额比例
sales_type=df.groupby('性别')['单价'].sum()
plt.figure(figsize=(8, 8))  
plt.pie(sales_type, labels=sales_type.index, autopct='%1.1f%%', startangle=140)
plt.title('男女消费金额比例')  
plt.axis('equal')
#个性词云图
text = ' '.join(user_behavior_products['具体商品'].astype(str))
wordcloud = WordCloud(font_path = r'C:/WINDOWS/Fonts/SIMHEI.TTF',width=800, height=400, background_color='white').generate(text)
# 显示词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
#plt.show()

user_behavior_products = pd.read_csv(r"E:\大三上\点宽杯\用户行为买了什么商品.csv")
user_behavior_products['时间'] = pd.to_datetime(user_behavior_products['时间'])
# 筛选2023/1/1到2024/9/30的数据
start_date = '2023-01-01'
end_date = '2024-09-30'
user_behavior_products_filtered = user_behavior_products[(user_behavior_products['时间'] >= start_date) & (user_behavior_products['时间'] <= end_date) & (user_behavior_products['行为'] == '购买')]
# 按季度分组并计算总销售额
sales_per_quarter = user_behavior_products_filtered.groupby(user_behavior_products_filtered['时间'].dt.to_period('Q'))['单价'].sum()
sales_per_quarter = sales_per_quarter.reset_index()
sales_per_quarter.columns = ['季度', '销售额']
pd.set_option('display.unicode.east_asian_width',True)
sales_per_quarter['季度'] = sales_per_quarter['季度'].astype(str)
plt.figure(figsize=(10, 6))  
plt.plot(sales_per_quarter['季度'], sales_per_quarter['销售额'], marker='o') 
plt.title('2023-2024年季度消费订单额')
plt.xlabel('季度')
plt.ylabel('消费订单额')
#按季度分组并计算每个季度的订单数量
order_counts_per_quarter = user_behavior_products.groupby(user_behavior_products['时间'].dt.to_period('Q')).size()
order_counts_per_quarter = order_counts_per_quarter.reset_index()
order_counts_per_quarter.columns = ['季度', '订单数']
order_counts_per_quarter['季度'] = order_counts_per_quarter['季度'].astype(str)
plt.figure(figsize=(10, 6))  
plt.plot(order_counts_per_quarter['季度'], order_counts_per_quarter['订单数'], marker='o') 
plt.title('2023-2024年季度消费订单数')
plt.xlabel('季度')
plt.ylabel('消费订单数')
#plt.show()
plt.rcParams['axes.unicode_minus'] = False
user_behavior_products = pd.read_csv(r"E:\大三上\点宽杯\用户行为买了什么商品.csv")
user_behavior_products['时间'] = pd.to_datetime(user_behavior_products['时间'])
# 筛选2023/1/1到2024/9/30的数据
start_date = '2023-01-01'
end_date = '2024-09-30'
user_behavior_products_filtered = user_behavior_products[(user_behavior_products['时间'] >= start_date) & (user_behavior_products['时间'] <= end_date) & (user_behavior_products['行为'] == '购买')]
#按月分组计算每个月的销售额
sales_per_month = user_behavior_products_filtered.groupby(user_behavior_products_filtered['时间'].dt.to_period('M'))['单价'].sum()
# 计算每个月的销售额增长率
sales_per_month_diff = sales_per_month.diff()
sales_growth_rate = (sales_per_month_diff / sales_per_month) * 100
# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(sales_growth_rate.index.astype(str), sales_growth_rate, marker='o', linestyle='-')
plt.title('每月销售额增长率')
plt.xlabel('月份')
plt.ylabel('增长率 (%)')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()
# 统计不同消费行为的个数
payment_methods = user_behavior_products['行为'].value_counts()
# 绘制饼图
plt.figure(figsize=(8, 8))  # 设置饼图大小
plt.pie(payment_methods,labels=payment_methods.index, autopct='%1.1f%%', startangle=140)
plt.title('消费手段分布')
plt.axis('equal')  
#plt.show()
# 统计支付方式
payment_methods = user_behavior_products['备注'].value_counts()
# 绘制饼图
plt.figure(figsize=(8, 8))  
plt.pie(payment_methods, labels=payment_methods.index, autopct='%1.1f%%', startangle=100)
plt.title('用户支付方式分布')  
plt.axis('equal')  # 使饼图为圆形
user_behavior_products=pd.read_csv(r"E:\大三上\点宽杯\用户行为买了什么商品.csv")
#plt.show()
promotions = pd.read_csv(r"E:\大三上\点宽杯\日期周末假期折扣量.csv")
promotions.rename(columns={'日期': '时间'}, inplace=True)
dw = pd.merge(user_behavior_products,promotions, on='时间', how='left')
# 根据是否是周末分组
weekend_data = dw[dw['周末'] == True]
weekday_data = dw[dw['周末'] == False]
# 根据是否是假期分组
holiday_data = dw[dw['假期'] == True]
non_holiday_data = dw[dw['假期'] == False]
weekend_actions = weekend_data['行为'].value_counts()
weekday_actions = weekday_data['行为'].value_counts()
holiday_actions = holiday_data['行为'].value_counts()
non_holiday_actions = non_holiday_data['行为'].value_counts()
# 创建周末和假期的行为分布表
weekend_table = pd.crosstab(weekend_data['行为'], weekend_data['周末'])
weekday_table = pd.crosstab(weekday_data['行为'], weekday_data['周末'])
holiday_table = pd.crosstab(holiday_data['行为'], holiday_data['假期'])
non_holiday_table = pd.crosstab(non_holiday_data['行为'], non_holiday_data['假期'])
# 进行卡方检验
chi2_weekend, p_weekend, dof_weekend, ex_weekend = stats.chi2_contingency(weekend_table)
chi2_weekday, p_weekday, dof_weekday, ex_weekday = stats.chi2_contingency(weekday_table)
chi2_holiday, p_holiday, dof_holiday, ex_holiday = stats.chi2_contingency(holiday_table)
chi2_non_holiday, p_non_holiday, dof_non_holiday, ex_non_holiday = stats.chi2_contingency(non_holiday_table)
print("周末卡方检验p值:", p_weekend)
print("工作日卡方检验p值:", p_weekday)
print("假期卡方检验p值:", p_holiday)
print("非假期卡方检验p值:", p_non_holiday)
# 绘制周末的用户行为分布
plt.figure(figsize=(10, 5))
sns.countplot(x='行为', data=weekend_data, hue='周末')
plt.title('用户周末行为')
# 绘制工作日的用户行为分布
plt.figure(figsize=(10, 5))
sns.countplot(x='行为', data=weekday_data, hue='周末')
plt.title('用户工作日行为')
# 绘制假期的用户行为分布
plt.figure(figsize=(10, 5))
sns.countplot(x='行为', data=holiday_data, hue='假期')
plt.title('用户假期行为')
# 绘制非假期的用户行为分布
plt.figure(figsize=(10, 5))
sns.countplot(x='行为', data=non_holiday_data, hue='假期')
plt.title('用户非假期行为')
plt.show()
