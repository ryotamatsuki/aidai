from dotenv import load_dotenv
load_dotenv()  # カレントディレクトリの.envファイルから環境変数をロード

import time
import os
import streamlit as st
import pandas as pd
import altair as alt
import re
from rapidfuzz import fuzz
import matplotlib.pyplot as plt

# 環境変数からAPIキーを取得し、存在しない場合はエラー表示
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("APIキーが設定されていません。環境変数（.envファイル等）を確認してください。")
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# google-generativeai ライブラリをインポート
try:
    import google.generativeai as palm
except ImportError:
    palm = None

def main():
    st.title('食事データダッシュボード')

    # カウントダウン表示（5秒間）
    status_area = st.empty()
    count_down_sec = 5
    for i in range(count_down_sec):
        status_area.write(f'{count_down_sec - i} sec left')
        time.sleep(1)
    status_area.write('Loading Dashboard...')

    # Gemini (Google Generative AI) API キーの設定
    if palm:
        palm.configure(api_key=os.environ["GOOGLE_API_KEY"])
    else:
        st.warning("google-generativeai ライブラリがインストールされていないため、Gemini チャット機能は利用できません。")

    # Geminiモデルの指定
    gemini_model = "gemini-2.0-flash-exp"

    # ---------------------
    # CSVデータの読み込み
    # ---------------------
    meal_details_csv = 'https://raw.githubusercontent.com/ryotamatsuki/aidai/refs/heads/main/meal_details.csv'
    df = pd.read_csv(meal_details_csv)
    df_nonzero = df[df['calories (kcal)'] != 0].copy()

    meal_behavior_csv = 'https://raw.githubusercontent.com/ryotamatsuki/aidai/refs/heads/main/imealbehavior_datai.csv'
    df_meal_behavior = pd.read_csv(meal_behavior_csv)

    # ---------------------
    # タブの作成（7つ）
    # ---------------------
    tab1, tab2, tab3, tab4, tab5, tab5_5, tab6 = st.tabs([
        "Raw Data",
        "Calorie Distribution",
        "Nutrition Totals",
        "Meal Summary by Timestamp",
        "Meal Behavior Stacked Bar",
        "Meal Action Percentage",
        "Meal Action Step Plots by Meal Timing"
    ])

    # ---------------------
    # タブ1: 生データの表示 + Gemini Chat (両CSV全体をRAGとして利用)
    # ---------------------
    with tab1:
        st.subheader("食事データの内容 (Raw Data)")
        st.dataframe(df_nonzero)

        st.write("### Data Chat (Gemini API)")
        if not palm:
            st.warning("Gemini API を利用できません。")
        else:
            user_question = st.text_area("データに関する質問を入力してください。")
            if st.button("送信", key="gemini_send"):
                if user_question.strip():
                    try:
                        # 両CSV全体をコンテキストとして渡す
                        context_text = (
                            "Meal Details Data:\n" + df.to_csv(index=False) +
                            "\n\nMeal Behavior Data:\n" + df_meal_behavior.to_csv(index=False)
                        )
                        combined_message = f"{context_text}\n\n質問: {user_question}"
                        
                        # モジュールレベルの関数としてpalm.chat()を直接呼び出す
                        response = palm.chat(
                            model=gemini_model,
                            messages=[{"role": "user", "content": combined_message}]
                        )
                        
                        if response:
                            st.write("#### 回答:")
                            st.write(response.text)
                        else:
                            st.write("Gemini API からのレスポンスがありませんでした。")
                    except Exception as e:
                        st.error(f"API呼び出しでエラーが発生しました: {e}")
                else:
                    st.info("質問を入力してください。")

    # ---------------------
    # タブ2: ファジーマッチング＋カロリー内訳（積み上げ棒グラフ＋総計テキスト付き）
    # ---------------------
    with tab2:
        st.subheader('タイムスタンプごとの Dish別 カロリー内訳')

        def basic_normalize(name):
            name = re.sub(r'\s*\(.*?\)', '', name)
            return name.strip().lower()

        df_nonzero['dish_norm'] = df_nonzero['dish'].apply(basic_normalize)
        unique_dishes = df_nonzero['dish_norm'].unique().tolist()

        threshold = 80
        def group_dishes(dish_list, threshold=80):
            groups = {}
            for dish in dish_list:
                found = False
                for rep in groups:
                    similarity = fuzz.ratio(dish, rep)
                    if similarity >= threshold:
                        groups[rep].append(dish)
                        found = True
                        break
                if not found:
                    groups[dish] = [dish]
            return groups

        groups = group_dishes(unique_dishes, threshold)
        mapping = {}
        for rep, members in groups.items():
            for m in members:
                mapping[m] = rep
        df_nonzero['dish_group'] = df_nonzero['dish_norm'].map(mapping)

        df_mapping_display = (
            df_nonzero[['dish', 'dish_group']]
            .drop_duplicates()
            .sort_values('dish')
            .rename(columns={'dish': 'Original Dish', 'dish_group': 'Grouped Dish'})
        )
        st.write("Dish → Group Mapping:")
        st.dataframe(df_mapping_display)

        if set(['timestamp', 'dish_group', 'calories (kcal)']).issubset(df_nonzero.columns):
            df_nonzero['timestamp'] = pd.to_datetime(df_nonzero['timestamp'])
            df_nonzero['timestamp_str'] = df_nonzero['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            grouped = df_nonzero.groupby(['timestamp_str', 'dish_group'], as_index=False)['calories (kcal)'].sum()
            sum_per_timestamp = grouped.groupby('timestamp_str', as_index=False)['calories (kcal)'].sum()
            st.write("各タイムスタンプの合計カロリー (kcal):")
            st.dataframe(sum_per_timestamp)

            bars = alt.Chart(grouped).mark_bar().encode(
                x=alt.X('timestamp_str:O', title='Timestamp', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('calories (kcal):Q', stack='zero', title='Total Calories (kcal)'),
                color=alt.Color('dish_group:N', title='Dish'),
                tooltip=['timestamp_str:N', 'dish_group:N', 'calories (kcal):Q']
            )
            text = alt.Chart(grouped).transform_aggregate(
                total='sum(calories (kcal))',
                groupby=['timestamp_str']
            ).mark_text(dy=-8, color='black').encode(
                x=alt.X('timestamp_str:O'),
                y=alt.Y('total:Q', stack='zero'),
                text=alt.Text('total:Q', format='.1f')
            )
            chart = alt.layer(bars, text).properties(width=alt.Step(80), height=400)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("CSVに 'timestamp', 'dish_group', 'calories (kcal)' のカラムが不足しています。")

    # ---------------------
    # タブ3: timestamp ごとの栄養素合計（テーブル表示）
    # ---------------------
    with tab3:
        required_cols = ['timestamp', 'calories (kcal)', 'fat (g)', 'carbohydrates (g)', 'protein (g)']
        if set(required_cols).issubset(df_nonzero.columns):
            st.subheader('栄養素の合計 (timestampごと)')
            df_nonzero['timestamp'] = pd.to_datetime(df_nonzero['timestamp'])
            grouped_nutrients = df_nonzero.groupby('timestamp')[['calories (kcal)', 'fat (g)', 'carbohydrates (g)', 'protein (g)']].sum().reset_index()
            st.dataframe(grouped_nutrients)
        else:
            st.write("必要なカラム（timestamp, calories (kcal), fat (g), carbohydrates (g), protein (g)）が存在しません。")

    # ---------------------
    # タブ4: 各栄養素の時系列推移（折れ線グラフのみ）
    # ---------------------
    with tab4:
        st.subheader("栄養素の時系列推移（折れ線グラフのみ）")
        if 'timestamp' in df_nonzero.columns:
            df_nonzero['timestamp'] = pd.to_datetime(df_nonzero['timestamp'])
            required_nutrients = ['fat (g)', 'carbohydrates (g)', 'protein (g)']
            if set(required_nutrients).issubset(df_nonzero.columns):
                nutrient_grouped = df_nonzero.groupby('timestamp', as_index=False)[required_nutrients].sum()
                nutrient_grouped['timestamp_str'] = nutrient_grouped['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                melted = nutrient_grouped.melt(
                    id_vars=['timestamp_str'],
                    value_vars=required_nutrients,
                    var_name='Nutrient',
                    value_name='Amount (g)'
                )
                nutrient_chart = alt.Chart(melted).mark_line(point=True).encode(
                    x=alt.X('timestamp_str:O', title='Timestamp'),
                    y=alt.Y('Amount (g):Q', title='Amount (g)'),
                    color=alt.Color('Nutrient:N', title='Nutrient'),
                    tooltip=['timestamp_str:N', 'Nutrient:N', 'Amount (g):Q']
                ).properties(width=800, height=400)
                st.altair_chart(nutrient_chart, use_container_width=True)
            else:
                st.write("必要な栄養素のカラムが不足しています。")
        else:
            st.write("CSVに 'timestamp' カラムがありません。")

    # ---------------------
    # タブ5: Meal Behavior Stacked Bar Chart (Matplotlib)
    # ---------------------
    with tab5:
        st.subheader("Meal Action Total Time (Stacked Bar Chart: Eat on Top)")
        behavior_csv = 'https://raw.githubusercontent.com/yourusername/yourrepo/main/path/to/imealbehavior_datai.csv'
        df_behavior = pd.read_csv(behavior_csv)
        df_behavior['meal_timing'] = pd.to_datetime(
            df_behavior['meal_timing'].astype(float) / 1000,
            unit='s',
            utc=True
        ).dt.tz_convert('Asia/Tokyo')
        df_behavior['timestamp'] = pd.to_datetime(df_behavior['timestamp'], format="%Y-%m-%d_%H-%M-%S.%f")
        df_behavior = df_behavior.sort_values(['meal_timing', 'timestamp'])
        duration_data = []
        for meal_timing, group in df_behavior.groupby('meal_timing'):
            group = group.sort_values('timestamp').copy()
            group_start = group['timestamp'].iloc[0]
            group_end = group['timestamp'].iloc[-1]
            group['segment'] = (group['meal_action'] != group['meal_action'].shift()).cumsum()
            segments = group.groupby('segment').agg(
                start=('timestamp', 'first'),
                meal_action=('meal_action', 'first')
            ).reset_index()
            segments['next_start'] = segments['start'].shift(-1)
            segments['duration'] = segments['next_start'] - segments['start']
            segments.loc[segments['duration'].isna(), 'duration'] = group_end - segments.loc[segments['duration'].isna(), 'start']
            segments['duration_seconds'] = segments['duration'].dt.total_seconds()
            for _, row in segments.iterrows():
                duration_data.append({
                    'meal_timing': meal_timing,
                    'meal_action': row['meal_action'].strip().lower(),
                    'duration': row['duration_seconds']
                })
        df_duration = pd.DataFrame(duration_data)
        df_pivot = df_duration.pivot_table(index='meal_timing', columns='meal_action', values='duration', aggfunc='sum').fillna(0)
        df_pivot.index = pd.to_datetime(df_pivot.index).strftime('%Y-%m-%d %H:%M:%S')
        meal_timings = df_pivot.index.tolist()
        fig, ax = plt.subplots(figsize=(8, 6))
        not_eat_durations = df_pivot.get('not eat', pd.Series(0, index=df_pivot.index))
        eat_durations = df_pivot.get('eat', pd.Series(0, index=df_pivot.index))
        total_durations = not_eat_durations + eat_durations
        bar_not_eat = ax.bar(meal_timings, not_eat_durations, color="#ff7f0e", label="Not Eat")
        bar_eat = ax.bar(meal_timings, eat_durations, bottom=not_eat_durations, color="#1f77b4", label="Eat")
        for i, mt in enumerate(meal_timings):
            tot = total_durations[mt]
            if tot > 0:
                not_eat_pct_val = (not_eat_durations[mt] / tot) * 100
                eat_pct_val = (eat_durations[mt] / tot) * 100
                ax.text(i, not_eat_durations[mt] / 2, f'{not_eat_durations[mt]:.1f} sec\n({not_eat_pct_val:.1f}%)',
                        ha='center', va='center', color='white', fontsize=10, fontweight='bold')
                ax.text(i, not_eat_durations[mt] + eat_durations[mt] / 2, f'{eat_durations[mt]:.1f} sec\n({eat_pct_val:.1f}%)',
                        ha='center', va='center', color='white', fontsize=10, fontweight='bold')
                ax.text(i, tot + tot * 0.05, f'Total: {tot:.1f} sec', ha='center', fontsize=10, fontweight='bold')
        max_val = total_durations.max()
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.25)
        ax.set_ylabel("Total Duration (seconds)")
        ax.set_title("Meal Action Duration by Meal Timing (Stacked Bar Chart: Eat on Top)")
        ax.legend([bar_eat, bar_not_eat], ["Eat", "Not Eat"], loc="upper right")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    # ---------------------
    # タブ5.5: 100%積み上げ棒グラフ（割合表示）の作成
    # ---------------------
    with tab5_5:
        st.subheader("Meal Action Percentage by Meal Timing (100% Stacked Bar Chart)")
        not_eat_durations = df_pivot.get('not eat', pd.Series(0, index=df_pivot.index))
        eat_durations = df_pivot.get('eat', pd.Series(0, index=df_pivot.index))
        total_durations = not_eat_durations + eat_durations
        not_eat_pct = (not_eat_durations / total_durations * 100).fillna(0)
        eat_pct = (eat_durations / total_durations * 100).fillna(0)
        meal_timings = df_pivot.index.tolist()
        fig, ax = plt.subplots(figsize=(8, 6))
        bar_not_eat_pct = ax.bar(meal_timings, not_eat_pct, color="#ff7f0e", label="Not Eat")
        bar_eat_pct = ax.bar(meal_timings, eat_pct, bottom=not_eat_pct, color="#1f77b4", label="Eat")
        for i, mt in enumerate(meal_timings):
            ne = not_eat_pct[mt]
            e = eat_pct[mt]
            tot = ne + e
            if tot > 0:
                ax.text(i, ne/2, f'{ne:.1f}%', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
                ax.text(i, ne + e/2, f'{e:.1f}%', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
                ax.text(i, tot + tot * 0.05, f'Total: 100%', ha='center', fontsize=10, fontweight='bold')
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Meal Action Percentage by Meal Timing (100% Stacked Bar Chart)")
        ax.set_ylim(0, 120)
        ax.legend([bar_eat_pct, bar_not_eat_pct], ["Eat", "Not Eat"], loc="upper right")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    # ---------------------
    # タブ6: Meal Action Step Plots by Meal Timing
    # ---------------------
    with tab6:
        st.subheader("Meal Action Step Plots by Meal Timing")
        behavior_csv = 'https://raw.githubusercontent.com/yourusername/yourrepo/main/path/to/imealbehavior_datai.csv'
        df_behavior = pd.read_csv(behavior_csv)
        df_behavior['timestamp'] = pd.to_datetime(df_behavior['timestamp'], format="%Y-%m-%d_%H-%M-%S.%f")
        df_behavior['state'] = df_behavior['meal_action'].apply(lambda x: 1 if x.strip().lower() == "eat" else 0)
        df_behavior = df_behavior.sort_values('timestamp')
        if 'meal_timing' not in df_behavior.columns:
            st.error("CSVファイルに 'meal_timing' カラムが存在しません。")
        else:
            for meal_timing, group in df_behavior.groupby('meal_timing'):
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.step(group['timestamp'], group['state'], where='post', label=f"Meal Action - {meal_timing}", linewidth=2)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["Not Eat (0)", "Eat (1)"])
                ax.set_xlabel("Timestamp")
                ax.set_ylabel("Meal State")
                ax.set_title(f"Meal Action Step Plot for Meal Timing: {meal_timing}")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend()
                st.pyplot(fig)

    st.balloons()

if __name__ == '__main__':
    main()
