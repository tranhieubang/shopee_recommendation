import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

# Cấu hình giao diện
st.set_page_config(page_title="Recommendation System", layout="wide")

# Đọc dữ liệu
path = 'C:\\Users\\LENOVO\\OneDrive\\Pictures\\Bigdata\\GUI_Project\\GUI_Cosine_similarity_model\\Products_ThoiTrangNam_raw.csv'
df_products = pd.read_csv(path)

# Chuẩn bị dữ liệu cho EDA
subcat_counts = df_products['sub_category'].value_counts().reset_index()
subcat_counts.columns = ['sub_category', 'count']
df_products['description_length'] = df_products['description'].fillna('').apply(len)

# Biểu đồ Top sub_category
fig_cat = px.bar(subcat_counts.sort_values('count', ascending=False).head(10),
                 x='sub_category', y='count',
                 title='Top 10 nhóm sản phẩm phổ biến',
                 text='count')

# Biểu đồ độ dài mô tả
fig_desc = px.histogram(df_products, x='description_length', nbins=30,
                        title="Phân bố độ dài mô tả sản phẩm")

# WordCloud
sampled_descriptions = df_products['description'].dropna().astype(str).sample(n=500, random_state=42)
text_data = ' '.join(sampled_descriptions)
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text_data)
fig_wc, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
plt.tight_layout()

# Sản phẩm trùng tên
top_products = df_products['product_name'].value_counts().reset_index()
top_products.columns = ['product_name', 'count']
top_duplicated = top_products[top_products['count'] > 1].head(10)
fig_dup = px.bar(top_duplicated.sort_values('count'),
                 x='count', y='product_name',
                 orientation='h',
                 title='\ud83d\udd01 Top 10 sản phẩm trùng tên nhiều nhất',
                 text='count',
                 labels={'count': 'Số lần xuất hiện', 'product_name': 'Tên sản phẩm'})

# Biểu đồ giá (nếu có)
if 'price' in df_products.columns:
    fig_price_dist = px.histogram(df_products, x='price', nbins=50,
                                   title='\ud83d\udcc8 Phân phối giá sản phẩm',
                                   labels={'price': 'Giá (VND)'})

    avg_price_by_subcat = df_products.groupby('sub_category')['price'].mean().sort_values(ascending=False).reset_index()
    fig_avg_price = px.bar(avg_price_by_subcat,
                           x='sub_category', y='price',
                           title='\ud83d\udcb0 Giá trung bình theo nhóm sản phẩm',
                           labels={'price': 'Giá trung bình (VND)', 'sub_category': 'Nhóm sản phẩm'},
                           text='price')
else:
    fig_price_dist = None
    fig_avg_price = None


# CSS căn giữa và giới hạn chiều rộng hợp lý
st.markdown("""
    <style>
    .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1200px;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Data Science")
st.write("## Recommendation System Project")

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Menu")
    menu = ["Overview", "Product Insights", "Recommendation"]
    choice = st.selectbox("Menu", menu)

    st.markdown("---")
    st.markdown("### 👤 Thông tin nhóm")
    st.markdown(
        """
        <div style='background-color: #f4f6fa; padding: 15px; border-radius: 10px; font-size: 15px; line-height: 1.6;'>
            🏅 <b>Thực hiện bởi:</b><br>
            <span style='font-weight: 500;'>Mai Hồng Hà & Trần Hiếu Băng</span><br><br>
            👩‍🏫 <b>Giảng viên:</b><br>
            Cô Khuất Thùy Phương<br><br>
            📅 <b>Ngày báo cáo:</b><br>
            20/04/2025
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr style='margin-top:20px; margin-bottom:10px;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:13px; color:gray;'>© 2025 Customer Segmentation Project</p>", unsafe_allow_html=True)


# TF-IDF cho mô tả
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_products['description'].fillna(""))

# Gợi ý theo mô tả
def get_recommendations_by_description(input_text, topn=5):
    input_vec = tfidf_vectorizer.transform([input_text])
    sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][:topn]
    return df_products.iloc[top_indices]

# Load batch cosine theo product_id
def load_cosine_batch(index, batch_size=1000):
    batch_file = f"cosine_batches/cosine_batch_{index // batch_size}.npy"
    batch = np.load(batch_file)
    row_index = index % batch_size
    return batch[row_index]

def get_recommendations(df, ma_san_pham, nums=5):
    matching_indices = df.index[df['product_id'] == ma_san_pham].tolist()
    if not matching_indices:
        st.warning(f"Không tìm thấy sản phẩm với ID: {ma_san_pham}")
        return pd.DataFrame()
    idx = matching_indices[0]
    sim_row = load_cosine_batch(idx)
    sim_scores = list(enumerate(sim_row))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums + 1]  # bỏ chính nó
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices]

# Hiển thị sản phẩm dạng card
def display_recommended_products(recommended_products, cols=3):
    st.markdown("""
        <style>
        .card-link {
            text-decoration: none;
        }
        .clickable-card {
            background-color: #fefefe;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 25px;
            transition: box-shadow 0.3s ease;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            text-align: center;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            min-height: 420px;
        }
        .clickable-card:hover {
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .product-title {
            font-size: 18px;
            font-weight: 600;
            margin: 12px 0 6px 0;
            color: #111;
            text-decoration: none;
        }
        .product-title:hover {
            color: #1a73e8;
        }
        .product-price {
            font-size: 16px;
            color: #007200;
            margin-bottom: 10px;
        }
        .product-desc {
            font-size: 14px;
            color: #444;
            line-height: 1.5;
            margin-top: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    for i in range(0, len(recommended_products), cols):
        col_set = st.columns(cols)
        for j, col in enumerate(col_set):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                product_name = product.get('product_name', 'Không tên')
                product_link = product.get('link', '#')
                price = product.get('price', 'Liên hệ')
                desc = product.get('description', '')
                desc = '' if pd.isna(desc) else str(desc)
                truncated = ' '.join(desc.split()[:25]) + '...'
                image_url = product.get('image', 'https://via.placeholder.com/150')

                with col:
                    st.markdown(f"""
                        <a href="{product_link}" target="_blank" class="card-link">
                            <div class="clickable-card">
                                <img src="{image_url}" style="width:100%; border-radius:10px; object-fit: cover; max-height: 180px;">
                                <div class="product-title">{product_name}</div>
                                <div class="product-price">{price} VND</div>
                                <div class="product-desc">📄 {truncated}</div>
                            </div>
                        </a>
                    """, unsafe_allow_html=True)

# Giao diện Recommendation
if choice == "Recommendation":
    #st.image("C:\\Users\\LENOVO\\OneDrive\\Pictures\\Bigdata\\GUI_Project\\GUI_Cosine_similarity_model\\hinh.png")

    st.subheader("🔀 Chọn phương thức đề xuất")
    method = st.radio("Chọn cách bạn muốn hệ thống gợi ý sản phẩm:", ["Theo mô tả", "Theo sản phẩm đã chọn"])

    if method == "Theo sản phẩm đã chọn":
        product_options = [(row['product_name'], row['product_id']) for _, row in df_products.head(10).iterrows()]
        selected_product = st.selectbox("Chọn sản phẩm", options=product_options, format_func=lambda x: x[0])
        selected_id = selected_product[1]
        nums = st.slider("Số lượng sản phẩm muốn đề xuất", min_value=1, max_value=10, value=3)

        st.write("Mã sản phẩm:", selected_id)
        selected_row = df_products[df_products['product_id'] == selected_id]

        if not selected_row.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_row['product_name'].values[0])
            desc = selected_row['description'].values[0]
            desc = '' if pd.isna(desc) else str(desc)
            st.write('##### Thông tin:')
            st.write(' '.join(desc.split()[:100]) + '...')
            st.write('##### Các sản phẩm liên quan:')
            recommendations = get_recommendations(df_products, selected_id, nums=nums)
            display_recommended_products(recommendations, cols=3)

    elif method == "Theo mô tả":
        nums = st.slider("Số lượng sản phẩm muốn đề xuất", min_value=1, max_value=10, value=3)
        user_input = st.text_input("Nhập mô tả sản phẩm bạn muốn tìm:")
        if user_input:
            st.write("##### Kết quả gợi ý từ mô tả:")
            recommended_by_desc = get_recommendations_by_description(user_input, topn=nums)
            display_recommended_products(recommended_by_desc, cols=3)
        else:
            st.info("Hãy nhập mô tả sản phẩm để nhận gợi ý.")

elif choice == "Overview":
    st.markdown("""
    <div style='font-size:22px; line-height:1.8; font-weight: 500; text-align: justify;'>
    🛍 <b>Shopee</b> là một hệ sinh thái thương mại <i>“all in one”</i>, trong đó <b>Shopee.vn</b> là một trong những nền tảng thương mại điện tử hàng đầu tại Việt Nam và khu vực Đông Nam Á.
    </div>
    """, unsafe_allow_html=True)

    st.image("C:\\Users\\LENOVO\\OneDrive\\Pictures\\Bigdata\\GUI_Project\\GUI_Cosine_similarity_model\\1-shopee-2024-mot-nam-nhin-lai20250111142551.png", width=800)

    st.markdown("<h3 style='color:#1f77b4;'>Vấn đề đặt ra:</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:17px; text-align: justify;'>
    Trong kho hàng hóa khổng lồ, việc giúp người dùng tìm được sản phẩm phù hợp là một thách thức lớn.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color:#d62728;'>Mục tiêu của dự án:</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:17px;'>
    <ul>
        <li>Xây dựng hệ thống <b>Recommendation System</b> giúp đề xuất sản phẩm phù hợp với người dùng</li>
        <li>Áp dụng cho một hoặc nhiều nhóm hàng hóa trên Shopee.vn</li>
        <li>Phân tích nội dung sản phẩm và gợi ý thông minh dựa trên tương đồng về mô tả và hành vi</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.success("📌 Hãy chuyển sang mục 'Recommendation' để bắt đầu trải nghiệm gợi ý sản phẩm thông minh!")

elif choice == "Product Insights":
    st.header("📊 Phân tích dữ liệu sản phẩm")

    st.markdown("### 🧮 Tổng quan dữ liệu sản phẩm")
    num_rows = st.slider("Chọn số dòng muốn xem", min_value=5, max_value=50, value=10)
    st.dataframe(df_products.head(num_rows))
    
    st.markdown("### 📄 Cấu trúc bảng dữ liệu (`.info()`)")
    import io
    buffer = io.StringIO()
    df_products.info(buf=buffer)
    st.text(buffer.getvalue())

    st.markdown("### 📌 Nhóm sản phẩm phổ biến")
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("### 📏 Phân bố độ dài mô tả")
    st.plotly_chart(fig_desc, use_container_width=True)

    st.markdown("### ☁️ WordCloud mô tả sản phẩm")
    st.pyplot(fig_wc)

    st.markdown("### 🧾 Sản phẩm trùng tên nhiều nhất")
    st.plotly_chart(fig_dup, use_container_width=True)
    st.info("🔄 Một số sản phẩm có tên giống nhau có thể đến từ nhiều shop khác nhau hoặc do sao chép nội dung.")

    if fig_price_dist and fig_avg_price:
        st.markdown("### 💸 Phân tích giá sản phẩm")
        st.plotly_chart(fig_price_dist, use_container_width=True)
        st.plotly_chart(fig_avg_price, use_container_width=True)
