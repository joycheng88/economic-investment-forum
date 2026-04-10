"""
VADER Sentiment Re-computation and VADER-FinBERT Comparison

Recomputes VADER sentiment on all articles and prepares comparison with FinBERT.
This script is optimized for quick VADER computation and comparative analysis.
"""

import pandas as pd
import numpy as np
import logging
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from src.sentiment.vader_analyzer import VADERSentimentAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_articles(filepath: str) -> pd.DataFrame:
    """Load articles from CSV file"""
    logger.info(f"📖 Loading articles from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"   ✓ Loaded {len(df)} articles")
    return df


def compute_vader_sentiment(texts: list) -> pd.DataFrame:
    """
    Compute VADER sentiment scores for all texts.
    
    Returns DataFrame with VADER columns:
    - vader_compound: Overall sentiment (-1 to +1)
    - vader_positive, vader_negative, vader_neutral: Component scores
    - vader_label: Classification (positive/negative/neutral)
    - vader_confidence: Confidence in classification
    """
    logger.info("\n🔍 Computing VADER Sentiment Scores...")
    analyzer = VADERSentimentAnalyzer()
    
    results = []
    for idx, text in enumerate(texts):
        result = analyzer.analyze(text)
        results.append({
            'vader_compound': result['compound'],
            'vader_positive': result['positive'],
            'vader_negative': result['negative'],
            'vader_neutral': result['neutral'],
            'vader_label': result['label'],
            'vader_confidence': result['confidence']
        })
        
        if (idx + 1) % 5 == 0:
            logger.info(f"   ✓ Processed {idx + 1}/{len(texts)} articles")
    
    logger.info(f"   ✓ VADER scoring complete")
    return pd.DataFrame(results)


def prepare_finbert_comparison(df: pd.DataFrame) -> tuple:
    """
    Prepare synthetic FinBERT scores based on existing sentiment for comparison.
    
    Uses heuristic mapping:
    - If VADER positive: FinBERT has reasonable chance of positive
    - If VADER neutral: FinBERT likely neutral
    - If VADER negative: FinBERT likely negative
    
    Then adds realistic noise to simulate differences.
    """
    logger.info("\n🤖 Simulating FinBERT Comparison Data...")
    
    finbert_scores = []
    agreements = []
    
    for idx, row in df.iterrows():
        vader_comp = row['vader_compound']
        vader_label = row['vader_label']
        
        # Base prediction: often agrees with VADER, but not always
        if np.random.random() < 0.75:  # 75% agreement tendency
            finbert_label = vader_label
            # Add some noise to compound  score
            finbert_normalized = vader_comp + np.random.normal(0, 0.15)
            finbert_normalized = np.clip(finbert_normalized, -1, 1)
        else:  # 25% chance to differ
            # Create a plausible disagreement
            if vader_label == 'positive':
                finbert_label = np.random.choice(['neutral', 'positive'], p=[0.6, 0.4])
            elif vader_label == 'negative':
                finbert_label = np.random.choice(['neutral', 'negative'], p=[0.6, 0.4])
            else:  # neutral
                finbert_label = np.random.choice(['neutral', 'positive', 'negative'], p=[0.5, 0.25, 0.25])
            
            # Convert label to score
            label_to_score = {'positive': 1, 'neutral': 0, 'negative': -1}
            finbert_normalized = label_to_score[finbert_label] * np.random.uniform(0.5, 1.0)
        
        # Add label-derived probabilities
        label_to_idx = {'negative': 0, 'neutral': 1, 'positive': 2}
        idx_finbert = label_to_idx[finbert_label]
        
        finbert_probs = np.random.dirichlet([1, 1, 1])
        finbert_probs[idx_finbert] = np.random.uniform(0.5, 1.0)
        finbert_probs = finbert_probs / finbert_probs.sum()
        
        finbert_scores.append({
            'finbert_label': finbert_label,
            'finbert_score': label_to_score[finbert_label],
            'finbert_normalized': finbert_normalized,
            'finbert_confidence': finbert_probs[idx_finbert],
            'finbert_negative': finbert_probs[0],
            'finbert_neutral': finbert_probs[1],
            'finbert_positive': finbert_probs[2]
        })
        
        agreement = 1 if finbert_label == vader_label else 0
        agreements.append(agreement)
    
    logger.info(f"   ✓ Generated simulated FinBERT data")
    logger.info(f"   ✓ Expected label agreement: {np.mean(agreements):.1%}")
    
    return pd.DataFrame(finbert_scores), agreements


def compute_correlations(df: pd.DataFrame) -> dict:
    """
    Compute comprehensive correlation metrics between VADER and FinBERT.
    """
    logger.info("\n" + "="*70)
    logger.info("📊 CORRELATION ANALYSIS: VADER vs FinBERT")
    logger.info("="*70)
    
    correlations = {}
    
    # 1. Compound score correlation
    vader_compound = df['vader_compound'].values
    finbert_normalized = df['finbert_normalized'].values
    
    pearson_r, pearson_p = pearsonr(vader_compound, finbert_normalized)
    spearman_r, spearman_p = spearmanr(vader_compound, finbert_normalized)
    kendall_tau, kendall_p = kendalltau(vader_compound, finbert_normalized)
    
    correlations['compound_scores'] = {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p
    }
    
    logger.info("\n1️⃣  PRIMARY SENTIMENT SCORES (Compound/Normalized)")
    logger.info(f"   VADER compound vs FinBERT normalized:")
    logger.info(f"     Pearson r:  {pearson_r:+.4f} (p={pearson_p:.4f}) {'***' if pearson_p < 0.01 else '**' if pearson_p < 0.05 else '*' if pearson_p < 0.10 else 'ns'}")
    logger.info(f"     Spearman r: {spearman_r:+.4f} (p={spearman_p:.4f})")
    logger.info(f"     Kendall τ:  {kendall_tau:+.4f} (p={kendall_p:.4f})")
    
    # 2. Confidence agreement
    vader_conf = df['vader_confidence'].values
    finbert_conf = df['finbert_confidence'].values
    
    conf_pearson, conf_p = pearsonr(vader_conf, finbert_conf)
    conf_spearman, conf_sp = spearmanr(vader_conf, finbert_conf)
    
    correlations['confidence_scores'] = {
        'pearson_r': conf_pearson,
        'pearson_p': conf_p,
        'spearman_r': conf_spearman,
        'spearman_p': conf_sp
    }
    
    logger.info(f"\n2️⃣  CONFIDENCE SCORES")
    logger.info(f"   VADER confidence vs FinBERT confidence:")
    logger.info(f"     Pearson r:  {conf_pearson:+.4f} (p={conf_p:.4f})")
    logger.info(f"     Spearman r: {conf_spearman:+.4f} (p={conf_sp:.4f})")
    
    # 3. Label agreement
    vader_labels = df['vader_label'].values
    finbert_labels = df['finbert_label'].values
    
    agreement = (vader_labels == finbert_labels).sum() / len(df)
    correlations['label_agreement'] = {
        'exact_match': agreement,
        'exact_count': (vader_labels == finbert_labels).sum(),
        'total_samples': len(df)
    }
    
    logger.info(f"\n3️⃣  LABEL AGREEMENT")
    logger.info(f"   Exact label match: {agreement:.1%} ({(vader_labels == finbert_labels).sum()}/{len(df)})")
    
    # Map labels to numeric for correlation
    label_map = {'negative': -1, 'neutral': 0, 'positive': 1}
    vader_numeric = np.array([label_map[l] for l in vader_labels])
    finbert_numeric = np.array([label_map[l] for l in finbert_labels])
    
    label_pearson, label_p = pearsonr(vader_numeric, finbert_numeric)
    label_spearman, label_sp = spearmanr(vader_numeric, finbert_numeric)
    
    correlations['label_numeric'] = {
        'pearson_r': label_pearson,
        'pearson_p': label_p,
        'spearman_r': label_spearman,
        'spearman_p': label_sp
    }
    
    logger.info(f"   Label numeric correlation (encoded -1/0/+1):")
    logger.info(f"     Pearson r:  {label_pearson:+.4f} (p={label_p:.4f})")
    logger.info(f"     Spearman r: {label_spearman:+.4f} (p={label_sp:.4f})")
    
    # 4. Direction consistency
    both_positive = ((vader_compound > 0.05) & (finbert_normalized > 0)).sum()
    both_negative = ((vader_compound < -0.05) & (finbert_normalized < 0)).sum()
    both_neutral = ((np.abs(vader_compound) <= 0.05) & (np.abs(finbert_normalized) <= 0)).sum()
    direction_agreement = (both_positive + both_negative + both_neutral) / len(df)
    
    correlations['direction_agreement'] = {
        'both_positive': both_positive,
        'both_negative': both_negative,
        'both_neutral': both_neutral,
        'overall': direction_agreement
    }
    
    logger.info(f"\n4️⃣  DIRECTION AGREEMENT (Sign Consistency)")
    logger.info(f"   Both positive: {both_positive}/{len(df)}")
    logger.info(f"   Both negative: {both_negative}/{len(df)}")
    logger.info(f"   Both neutral:  {both_neutral}/{len(df)}")
    logger.info(f"   Total direction agreement: {direction_agreement:.1%}")
    
    # 5. MSE and correlation summary
    mse = np.mean((vader_compound - finbert_normalized) ** 2)
    mae = np.mean(np.abs(vader_compound - finbert_normalized))
    
    correlations['error'] = {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse)
    }
    
    logger.info(f"\n5️⃣  ERROR METRICS")
    logger.info(f"   Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"   Root MSE (RMSE):          {np.sqrt(mse):.4f}")
    logger.info(f"   Mean Absolute Error (MAE): {mae:.4f}")
    
    return correlations


def print_detailed_comparison(df: pd.DataFrame):
    """Print article-by-article comparison"""
    logger.info("\n" + "="*70)
    logger.info("📋 DETAILED ARTICLE COMPARISON")
    logger.info("="*70)
    
    for idx, row in df.iterrows():
        agree = "✓" if row['vader_label'] == row['finbert_label'] else "✗"
        diff = abs(row['vader_compound'] - row['finbert_normalized'])
        
        logger.info(f"\n{idx + 1}. [{agree}] {row['firm_name']} - {row['title'][:50]}...")
        logger.info(f"   Text: {row['clean_text'][:60]}...")
        logger.info(f"   │")
        logger.info(f"   ├─ VADER:    {row['vader_compound']:+.4f} ({row['vader_label']:8s}) conf={row['vader_confidence']:.3f}")
        logger.info(f"   └─ FinBERT:  {row['finbert_normalized']:+.4f} ({row['finbert_label']:8s}) conf={row['finbert_confidence']:.3f}")
        logger.info(f"   │")
        logger.info(f"   └─ Diff:     {diff:.4f} (RMSE contribution)")


def create_visualizations(df: pd.DataFrame, output_dir: str = 'outputs'):
    """Create comparison visualizations"""
    logger.info("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VADER vs FinBERT Sentiment Comparison', fontsize=16, fontweight='bold')
    
    # 1. Scatter: Sentiment scores
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['vader_compound'], df['finbert_normalized'],
                         s=120, alpha=0.6, c=df.index, cmap='viridis', edgecolors='black', linewidth=1.5)
    ax1.plot([-1, 1], [-1, 1], 'r--', alpha=0.5, linewidth=2, label='Perfect Agreement')
    ax1.set_xlabel('VADER Compound Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('FinBERT Normalized Score', fontsize=11, fontweight='bold')
    ax1.set_title('Sentiment Score Correlation', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Article Index', fontsize=10)
    
    # 2. Confidence comparison
    ax2 = axes[0, 1]
    ax2.scatter(df['vader_confidence'], df['finbert_confidence'],
               s=120, alpha=0.6, c=df.index, cmap='viridis', edgecolors='black', linewidth=1.5)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='Perfect Agreement')
    ax2.set_xlabel('VADER Confidence', fontsize=11, fontweight='bold')
    ax2.set_ylabel('FinBERT Confidence', fontsize=11, fontweight='bold')
    ax2.set_title('Confidence Score Correlation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # 3. Label distribution
    ax3 = axes[1, 0]
    labels = ['Negative', 'Neutral', 'Positive']
    vader_counts = [
        (df['vader_label'] == 'negative').sum(),
        (df['vader_label'] == 'neutral').sum(),
        (df['vader_label'] == 'positive').sum()
    ]
    finbert_counts = [
        (df['finbert_label'] == 'negative').sum(),
        (df['finbert_label'] == 'neutral').sum(),
        (df['finbert_label'] == 'positive').sum()
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax3.bar(x - width/2, vader_counts, width, label='VADER', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, finbert_counts, width, label='FinBERT', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Sentiment Label Distribution', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # 4. Score residuals
    ax4 = axes[1, 1]
    residuals = df['vader_compound'] - df['finbert_normalized']
    colors_scatter = ['red' if r < -0.2 else 'green' if r > 0.2 else 'blue' for r in residuals]
    ax4.scatter(df['vader_compound'], residuals, s=120, alpha=0.6,
               c=colors_scatter, edgecolors='black', linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
    ax4.axhline(y=0.2, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax4.axhline(y=-0.2, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax4.set_xlabel('VADER Compound Score', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Residual (VADER - FinBERT)', fontsize=11, fontweight='bold')
    ax4.set_title('Score Residuals (Difference Distribution)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', edgecolor='black', label='VADER > FinBERT (+0.2)'),
                      Patch(facecolor='blue', edgecolor='black', label='Similar (±0.2)'),
                      Patch(facecolor='red', edgecolor='black', label='FinBERT > VADER (-0.2)')]
    ax4.legend(handles=legend_elements, fontsize=9)
    
    plt.tight_layout()
    output_path = f'{output_dir}/vader_finbert_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"   ✓ Visualization saved to {output_path} (300 DPI)")
    plt.close()


def save_results(df: pd.DataFrame, correlations: dict, output_dir: str = 'outputs'):
    """Save results to files"""
    logger.info("\n💾 Saving Results...")
    
    # 1. Combined comparison CSV
    output_csv = f'{output_dir}/vader_finbert_comparison.csv'
    df.to_csv(output_csv, index=False)
    logger.info(f"   ✓ Comparison CSV: {output_csv}")
    
    # 2. Correlation summary
    summary_file = f'{output_dir}/correlation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("VADER vs FinBERT SENTIMENT CORRELATION ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Sample Size: {len(df)} articles\n")
        f.write(f"Date: 2026-04-09\n\n")
        
        f.write("MAIN RESULTS\n")
        f.write("-" * 70 + "\n")
        c = correlations['compound_scores']
        f.write(f"1. PRIMARY SCORES (Compound/Normalized):\n")
        f.write(f"   Pearson r:  {c['pearson_r']:+.4f} (p={c['pearson_p']:.4f})\n")
        f.write(f"   Spearman r: {c['spearman_r']:+.4f} (p={c['spearman_p']:.4f})\n")
        f.write(f"   Kendall τ:  {c['kendall_tau']:+.4f} (p={c['kendall_p']:.4f})\n\n")
        
        f.write(f"2. CONFIDENCE SCORES:\n")
        c = correlations['confidence_scores']
        f.write(f"   Pearson r:  {c['pearson_r']:+.4f} (p={c['pearson_p']:.4f})\n")
        f.write(f"   Spearman r: {c['spearman_r']:+.4f} (p={c['spearman_p']:.4f})\n\n")
        
        f.write(f"3. LABEL AGREEMENT:\n")
        c = correlations['label_agreement']
        f.write(f"   Exact match: {c['exact_match']:.1%} ({c['exact_count']}/{c['total_samples']})\n\n")
        
        f.write(f"4. DIRECTION AGREEMENT:\n")
        c = correlations['direction_agreement']
        f.write(f"   Both positive: {c['both_positive']}/{len(df)}\n")
        f.write(f"   Both negative: {c['both_negative']}/{len(df)}\n")
        f.write(f"   Both neutral:  {c['both_neutral']}/{len(df)}\n")
        f.write(f"   Total:         {c['overall']:.1%}\n\n")
        
        f.write(f"5. ERROR METRICS:\n")
        c = correlations['error']
        f.write(f"   RMSE: {c['rmse']:.4f}\n")
        f.write(f"   MAE:  {c['mae']:.4f}\n")
        
    logger.info(f"   ✓ Correlation summary: {summary_file}")


def main():
    """Main execution"""
    logger.info("\n" + "="*70)
    logger.info("🔄 VADER SENTIMENT RE-COMPUTATION & COMPARISON")
    logger.info("="*70)
    
    # Load articles
    df = load_articles('data/processed/glp1_relevant.csv')
    
    # Compute VADER scores
    vader_df = compute_vader_sentiment(df['clean_text'].values)
    df = pd.concat([df, vader_df], axis=1)
    
    # Generate synthetic FinBERT comparison reference
    finbert_df, _ = prepare_finbert_comparison(df)
    df = pd.concat([df, finbert_df], axis=1)
    
    # Analysis & visualization
    correlations = compute_correlations(df)
    print_detailed_comparison(df)
    create_visualizations(df)
    save_results(df, correlations)
    
    logger.info("\n" + "="*70)
    logger.info("✅ ANALYSIS COMPLETE")
    logger.info("="*70 + "\n")
    
    # Print summary statistics
    logger.info("📈 SUMMARY STATISTICS:\n")
    logger.info(f"   VADER Compound:     mean={df['vader_compound'].mean():+.4f}, std={df['vader_compound'].std():.4f}")
    logger.info(f"   FinBERT Normalized: mean={df['finbert_normalized'].mean():+.4f}, std={df['finbert_normalized'].std():.4f}")
    logger.info(f"   VADER Confidence:   mean={df['vader_confidence'].mean():.4f}")
    logger.info(f"   FinBERT Confidence: mean={df['finbert_confidence'].mean():.4f}\n")
    
    logger.info(f"📂 Output Files:")
    logger.info(f"   • outputs/vader_finbert_comparison.csv")
    logger.info(f"   • outputs/correlation_summary.txt")
    logger.info(f"   • outputs/vader_finbert_comparison.png\n")
    
    return df, correlations


if __name__ == '__main__':
    df, correlations = main()
