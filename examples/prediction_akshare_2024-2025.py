import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "yuce"

# Add project path so custom modules can be imported
sys.path.append("../src")
from kronos import Kronos, KronosTokenizer, KronosPredictor

# Configure fonts
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def ensure_output_directory(output_dir):
    """Ensure the output directory exists, creating it if needed"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory: {output_dir}")
    return output_dir


def prepare_stock_data(csv_file_path, stock_code):
    """
    准备股票数据，转换为Kronos模型需要的格式

    参数:
    csv_file_path: CSV文件路径
    stock_code: 股票代码，用于显示信息

    返回:
    df: 处理后的DataFrame
    """
    print(f"Loading and preprocessing data for stock {stock_code}...")

    # 读取CSV文件
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')

    # 检查数据列名并重命名为标准格式
    column_mapping = {
        '日期': 'timestamps',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '成交额': 'amount'
    }

    # 只重命名存在的列
    actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=actual_mapping)

    # 确保时间戳列存在并转换为datetime格式
    if 'timestamps' not in df.columns:
        # 如果数据有日期索引，重置索引
        if df.index.name == '日期':
            df = df.reset_index()
            df = df.rename(columns={'日期': 'timestamps'})

    df['timestamps'] = pd.to_datetime(df['timestamps'])

    # 按时间排序
    df = df.sort_values('timestamps').reset_index(drop=True)

    print(f"✅ Data loaded, {len(df)} records in total")
    print(f"Time range: {df['timestamps'].min()} to {df['timestamps'].max()}")
    print(f"Data columns: {df.columns.tolist()}")

    return df


def calculate_prediction_parameters(df, target_days=100):
    """
    根据目标预测天数计算合适的参数

    参数:
    df: 股票数据DataFrame
    target_days: 目标预测天数（自然日）

    返回:
    lookback: 回看期数
    pred_len: 预测期数
    """
    # 计算平均交易日数量（考虑节假日）
    total_days = (df['timestamps'].max() - df['timestamps'].min()).days
    trading_days = len(df)
    trading_ratio = trading_days / total_days if total_days > 0 else 0.7  # 交易日比例

    # 计算目标预测的交易日数量
    pred_trading_days = int(target_days * trading_ratio)

    # 设置回看期数为预测期数的2-3倍，但不超过数据总量的70%
    max_lookback = int(len(df) * 0.7)
    lookback = min(pred_trading_days * 2, max_lookback, len(df) - pred_trading_days)
    pred_len = min(pred_trading_days, len(df) - lookback)

    print(f"📊 Parameter calculation:")
    print(f"  Target forecast horizon: {target_days} calendar days")
    print(f"  Estimated trading days: {pred_trading_days}")
    print(f"  Lookback window (lookback): {lookback}")
    print(f"  Prediction length (pred_len): {pred_len}")

    return lookback, pred_len


def generate_future_dates_with_holidays(last_date, pred_len):
    """
    生成未来的交易日日期，考虑中国节假日

    参数:
    last_date: 最后一个历史数据的日期
    pred_len: 预测期数

    返回:
    future_dates: 未来的交易日日期列表
    """
    # 中国主要节假日（需要根据实际情况调整）
    holidays_2025 = [
        # 2025年国庆节假期（通常为10月1日-10月8日）
        datetime(2025, 10, 1), datetime(2025, 10, 2), datetime(2025, 10, 3),
        datetime(2025, 10, 4), datetime(2025, 10, 5), datetime(2025, 10, 6),
        datetime(2025, 10, 7), datetime(2025, 10, 8),  # 添加10月8日
        # 周末调休等可以根据需要添加
    ]

    future_dates = []
    current_date = last_date + timedelta(days=1)

    while len(future_dates) < pred_len:
        # 如果是工作日（周一到周五）且不是节假日
        if current_date.weekday() < 5 and current_date not in holidays_2025:
            future_dates.append(current_date)
        current_date += timedelta(days=1)

    print(f"📅 Generated future trading days: {len(future_dates)} in total")
    print(f"   Start date: {future_dates[0].strftime('%Y-%m-%d')}")
    print(f"   End date: {future_dates[-1].strftime('%Y-%m-%d')}")

    # Show holiday information
    holiday_count = sum(1 for date in holidays_2025 if date > last_date)
    print(f"   Holidays included: {holiday_count}")

    return future_dates[:pred_len]


def plot_prediction_with_details(kline_df, pred_df, future_dates, stock_code="002354", stock_name="Stock", pred_len=100,
                                 output_dir="."):
    """
    绘制详细的预测结果图表 - 优化版，图表更大更清晰

    参数:
    kline_df: 历史K线数据
    pred_df: 预测数据
    future_dates: 未来日期列表
    stock_code: 股票代码
    stock_name: 股票名称
    pred_len: 预测期数
    output_dir: 输出目录
    """
    # 确保输出目录存在
    ensure_output_directory(output_dir)

    # 确保数据长度一致
    min_len = min(len(pred_df), len(future_dates))
    pred_df = pred_df.iloc[:min_len]
    future_dates = future_dates[:min_len]

    # 设置预测数据的索引为未来日期
    pred_df.index = future_dates

    # 准备价格数据
    sr_close = kline_df.set_index('timestamps')['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Historical'
    sr_pred_close.name = "Predicted"

    # 准备成交量数据
    sr_volume = kline_df.set_index('timestamps')['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Historical'
    sr_pred_volume.name = "Predicted"

    # 合并数据
    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    # 创建更大的图表
    fig = plt.figure(figsize=(18, 14))

    # 使用GridSpec创建更灵活的布局
    gs = plt.GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1])

    ax1 = fig.add_subplot(gs[0])  # 价格图表
    ax2 = fig.add_subplot(gs[1])  # 成交量图表
    ax3 = fig.add_subplot(gs[2])  # 价格变动图表

    # 1. 价格图表 - 更大更清晰
    # 只显示最近200个交易日的历史数据，避免图表过于拥挤
    recent_history = close_df['Historical'].iloc[-min(200, len(close_df['Historical'])):]
    ax1.plot(recent_history.index, recent_history.values, label='Historical Price', color='#1f77b4', linewidth=2.5, alpha=0.9)
    ax1.plot(close_df['Predicted'].index, close_df['Predicted'].values, label='Predicted Price',
             color='#ff7f0e', linewidth=2.5, linestyle='-', marker='o', markersize=3)

    # 添加预测起始点的标记
    prediction_start_date = close_df['Predicted'].index[0] if len(close_df['Predicted']) > 0 else close_df.index[-1]
    prediction_start_price = close_df['Historical'].iloc[-1]
    ax1.axvline(x=prediction_start_date, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.annotate('Forecast start', xy=(prediction_start_date, prediction_start_price),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.set_ylabel('Close Price (CNY)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{stock_name}({stock_code}) Stock Price Forecast - Next {pred_len} Trading Days',
                  fontsize=16, fontweight='bold', pad=20)

    # 设置x轴日期格式
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 设置y轴格式
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    # 2. 成交量图表 - 优化显示
    # 只显示预测期的成交量
    pred_volumes = volume_df['Predicted'].dropna()
    if len(pred_volumes) > 0:
        ax2.bar(pred_volumes.index, pred_volumes.values,
                alpha=0.7, color='#ff7f0e', label='Predicted Volume', width=0.8)

    ax2.set_ylabel('Volume (lots)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 设置x轴标签
    if len(pred_volumes) > 0:
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 3. 价格变动图表 - 优化显示
    if len(close_df['Predicted']) > 0:
        price_change = close_df['Predicted'] - close_df['Historical'].iloc[-1]
        colors = ['green' if x >= 0 else 'red' for x in price_change]

        # 每5个交易日显示一个标签，避免过于拥挤
        bars = ax3.bar(range(len(price_change)), price_change, alpha=0.8, color=colors)

        # 在关键点添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if i % 10 == 0 or i == len(bars) - 1 or abs(height) > price_change.std():  # 每10天或最后一天或显著波动
                ax3.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:+.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=8, fontweight='bold')

        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    ax3.set_ylabel('Price Change (CNY)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Trading Day', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 设置x轴标签
    if len(price_change) > 0:
        # 每10个交易日显示一个标签
        xticks_positions = list(range(0, len(price_change), max(1, len(price_change) // 10)))
        if len(price_change) - 1 not in xticks_positions:
            xticks_positions.append(len(price_change) - 1)
        ax3.set_xticks(xticks_positions)
        ax3.set_xticklabels([f'D{i + 1}' for i in xticks_positions])

    # 添加详细的统计信息框
    if len(close_df['Predicted']) > 0 and not np.isnan(close_df['Historical'].iloc[-1]):
        pred_stats = {
            'Stock Code': stock_code,
            'Stock Name': stock_name,
            'Current Price': f"{close_df['Historical'].iloc[-1]:.2f} CNY",
            'Predicted End Price': f"{close_df['Predicted'].iloc[-1]:.2f} CNY",
            'Predicted Change': f"{(close_df['Predicted'].iloc[-1] / close_df['Historical'].iloc[-1] - 1) * 100:+.2f}%",
            'Predicted High': f"{close_df['Predicted'].max():.2f} CNY",
            'Predicted Low': f"{close_df['Predicted'].min():.2f} CNY",
            'Predicted Volatility': f"{close_df['Predicted'].std():.2f} CNY",
            'Forecast Start Date': f"{close_df['Predicted'].index[0].strftime('%Y-%m-%d')}",
            'Forecast End Date': f"{close_df['Predicted'].index[-1].strftime('%Y-%m-%d')}",
            'Forecast Trading Days': f"{len(close_df['Predicted'])} days"
        }

        stats_text = "\n".join([f"{k}: {v}" for k, v in pred_stats.items()])
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                 verticalalignment='bottom')

    plt.tight_layout()

    # 保存高分辨率图片到指定目录
    chart_filename = os.path.join(output_dir, f'{stock_code}_prediction_chart.png')
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Forecast chart saved: {chart_filename}")

    plt.show()

    return close_df, volume_df


def generate_prediction_report(close_df, volume_df, pred_df, future_dates, stock_code="002354", stock_name="Stock",
                               output_dir="."):
    """
    生成预测报告
    """
    # 确保输出目录存在
    ensure_output_directory(output_dir)

    print(f"\n{'=' * 70}")
    print(f"📊 {stock_name}({stock_code}) Stock Forecast Report")
    print(f"{'=' * 70}")

    if len(close_df['Predicted']) == 0 or np.isnan(close_df['Historical'].iloc[-1]):
        print("❌ No valid prediction data available to generate the report")
        return

    # 确保所有数组长度一致
    min_len = min(len(close_df['Predicted']), len(volume_df['Predicted']), len(future_dates))

    # 基本统计
    historical_close = close_df['Historical'].iloc[-1]
    predicted_close = close_df['Predicted'].iloc[-1]
    price_change_pct = (predicted_close / historical_close - 1) * 100

    print(f"🔮 Forecast overview:")
    print(f"   Current price: {historical_close:.2f} CNY")
    print(f"   Forecast end price: {predicted_close:.2f} CNY")
    print(f"   Forecast change: {price_change_pct:+.2f}%")
    print(f"   Forecast horizon: {min_len} trading days")
    print(
        f"   Forecast time range: {future_dates[0].strftime('%Y-%m-%d')} to {future_dates[min_len - 1].strftime('%Y-%m-%d')}")

    print(f"\n📈 Price forecast statistics:")
    print(f"   Forecast high: {close_df['Predicted'].max():.2f} CNY")
    print(f"   Forecast low: {close_df['Predicted'].min():.2f} CNY")
    print(f"   Forecast mean: {close_df['Predicted'].mean():.2f} CNY")
    print(f"   Price volatility: {close_df['Predicted'].std():.2f} CNY")

    print(f"\n📊 Volume forecast statistics:")
    print(f"   Forecast mean volume: {volume_df['Predicted'].mean():,.0f} lots")
    print(f"   Forecast max volume: {volume_df['Predicted'].max():,.0f} lots")
    print(f"   Forecast min volume: {volume_df['Predicted'].min():,.0f} lots")

    # Save detailed prediction data to the output directory - keep all arrays the same length
    prediction_details = pd.DataFrame({
        'Date': future_dates[:min_len],
        'Predicted Close': close_df['Predicted'].values[:min_len],
        'Predicted Volume': volume_df['Predicted'].values[:min_len],
        'Price Change (CNY)': (close_df['Predicted'].values[:min_len] - historical_close),
        'Price Change (%)': ((close_df['Predicted'].values[:min_len] / historical_close - 1) * 100)
    })

    prediction_file = os.path.join(output_dir, f'{stock_code}_detailed_predictions.csv')
    prediction_details.to_csv(prediction_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 Detailed prediction data saved: {prediction_file}")


def main(stock_code="002354", stock_name="Tianyu Shuke", data_dir="./data", pred_days=100, output_dir="./output"):
    """
    主函数：执行股票价格预测

    参数:
    stock_code: 股票代码
    stock_name: 股票名称
    data_dir: 数据文件目录
    pred_days: 预测天数（自然日）
    output_dir: 输出文件目录
    """
    # 构建数据文件路径
    csv_file_path = os.path.join(data_dir, f"{stock_code}_stock_data.csv")

    print(f"🎯 Starting stock price forecast for {stock_name}({stock_code})")
    print("=" * 70)
    print(f"Data file: {csv_file_path}")
    print(f"Forecast horizon: {pred_days} calendar days")
    print(f"Output directory: {output_dir}")

    # Check whether the data file exists
    if not os.path.exists(csv_file_path):
        print(f"❌ Data file not found: {csv_file_path}")
        print("Please run the data collection script first to generate the stock data file")
        return

    # 确保输出目录存在
    ensure_output_directory(output_dir)

    try:
        # 1. 加载模型和分词器
        print("\nStep 1: Loading Kronos model and tokenizer...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        print("✅ Model loaded")

        # 2. Instantiate the predictor
        print("Step 2: Initializing predictor...")
        predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
        print("✅ Predictor initialized")

        # 3. Prepare the data
        print("Step 3: Preparing stock data...")
        df = prepare_stock_data(csv_file_path, stock_code)

        # 4. Compute prediction parameters
        print("Step 4: Computing prediction parameters...")
        lookback, pred_len = calculate_prediction_parameters(df, target_days=pred_days)

        if pred_len <= 0:
            print("❌ Insufficient data, unable to make a prediction")
            return

        print(f"✅ Final parameters - lookback: {lookback}, pred_len: {pred_len}")

        # 5. Prepare input data
        print("Step 5: Preparing input data...")
        # 使用最新的数据作为输入
        x_df = df.loc[-lookback:, ['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
        x_timestamp = df.loc[-lookback:, 'timestamps'].reset_index(drop=True)

        # 生成未来日期（考虑节假日）
        last_historical_date = df['timestamps'].iloc[-1]
        future_dates = generate_future_dates_with_holidays(last_historical_date, pred_len)

        print(f"Input data shape: {x_df.shape}")
        print(f"Historical data time range: {x_timestamp.iloc[0]} to {x_timestamp.iloc[-1]}")
        print(f"Forecast time range: {future_dates[0]} to {future_dates[-1]}")

        # 6. Run the prediction
        print("Step 6: Running price prediction...")
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=pd.Series(future_dates),  # 使用未来日期作为预测时间戳
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=True
        )

        print("✅ Prediction complete")

        # 7. Show prediction results
        print("\nStep 7: Showing prediction results...")
        print("First 5 rows of predicted data:")
        # 确保预测数据长度与未来日期一致
        min_len = min(len(pred_df), len(future_dates))
        pred_df = pred_df.iloc[:min_len]
        pred_df.index = future_dates[:min_len]
        print(pred_df.head())

        # 8. 可视化结果
        print("Step 8: Generating visualization charts...")
        # 使用最后一部分历史数据和预测数据
        kline_df = df.loc[-lookback:].reset_index(drop=True)
        close_df, volume_df = plot_prediction_with_details(kline_df, pred_df, future_dates, stock_code, stock_name,
                                                           pred_len, output_dir)

        # 9. 生成预测报告
        print("Step 9: Generating prediction report...")
        generate_prediction_report(close_df, volume_df, pred_df, future_dates, stock_code, stock_name, output_dir)

        print(f"\n🎉 Forecast complete for {stock_name}({stock_code})!")
        print("Generated files:")
        print(f"  📊 {os.path.join(output_dir, stock_code + '_prediction_chart.png')} - forecast chart")
        print(f"  📋 {os.path.join(output_dir, stock_code + '_detailed_predictions.csv')} - detailed predictions")

        # 显示预测总结
        if len(close_df['Predicted']) > 0 and not np.isnan(close_df['Historical'].iloc[-1]):
            print(f"\n📈 Forecast summary:")
            historical_price = close_df['Historical'].iloc[-1]
            predicted_price = close_df['Predicted'].iloc[-1]
            change_pct = (predicted_price / historical_price - 1) * 100

            print(f"  Current price: {historical_price:.2f} CNY")
            print(f"  Predicted price: {predicted_price:.2f} CNY")
            print(f"  Expected change: {change_pct:+.2f}%")
            print(
                f"  Forecast window: {future_dates[0].strftime('%Y-%m-%d')} to {future_dates[min_len - 1].strftime('%Y-%m-%d')}")

            if change_pct > 10:
                print(f"  🚀 Model forecasts a strong bullish move over the next {pred_len} trading days (+{change_pct:.1f}%)")
            elif change_pct > 5:
                print(f"  📈 Model forecasts a bullish move over the next {pred_len} trading days (+{change_pct:.1f}%)")
            elif change_pct > 0:
                print(f"  ↗️ Model forecasts a slight rise over the next {pred_len} trading days (+{change_pct:.1f}%)")
            elif change_pct > -5:
                print(f"  ↘️ Model forecasts a slight decline over the next {pred_len} trading days ({change_pct:.1f}%)")
            elif change_pct > -10:
                print(f"  📉 Model forecasts a bearish move over the next {pred_len} trading days ({change_pct:.1f}%)")
            else:
                print(f"  🔻 Model forecasts a strong bearish move over the next {pred_len} trading days ({change_pct:.1f}%)")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


# 使用方法说明
if __name__ == "__main__":
    """
    股票预测工具 - 支持多股票预测

    使用方法：
    修改下面的 STOCK_CONFIG 来预测不同的股票
    """

    # ==================== 在这里修改股票配置 ====================
    STOCK_CONFIG = {
        "stock_code": "300418",  # stock code
        "stock_name": "Kunlun Tech",  # stock name
        "data_dir": str(DEFAULT_DATA_DIR),  # data directory
        "pred_days": 100,  # forecast 100 calendar days
        "output_dir": str(DEFAULT_OUTPUT_DIR)  # output directory
    }

    # 其他股票配置示例：
    # STOCK_CONFIG = {"stock_code": "000001", "stock_name": "Ping An Bank", "data_dir": str(DEFAULT_DATA_DIR), "pred_days": 100, "output_dir": str(DEFAULT_OUTPUT_DIR)}
    # STOCK_CONFIG = {"stock_code": "600036", "stock_name": "China Merchants Bank", "data_dir": str(DEFAULT_DATA_DIR), "pred_days": 100, "output_dir": str(DEFAULT_OUTPUT_DIR)}
    # STOCK_CONFIG = {"stock_code": "300750", "stock_name": "CATL", "data_dir": str(DEFAULT_DATA_DIR), "pred_days": 100, "output_dir": str(DEFAULT_OUTPUT_DIR)}
    # =========================================================

    print("🤖 Smart Stock Forecast Tool")
    print("=" * 70)
    print(f"Current stock: {STOCK_CONFIG['stock_name']}({STOCK_CONFIG['stock_code']})")
    print(f"Data directory: {STOCK_CONFIG['data_dir']}")
    print(f"Forecast horizon: {STOCK_CONFIG['pred_days']} calendar days")
    print(f"Output directory: {STOCK_CONFIG['output_dir']}")
    print()

    # Run the main routine
    main(**STOCK_CONFIG)

    print(f"\n💡 Tip: to forecast another stock, edit the STOCK_CONFIG variable in this script")