"""
Compare VADER and FinBERT Sentiment Scores

Loads articles with existing VADER scores, computes FinBERT scores,
and performs detailed correlation analysis.
"""

import pandas as pd
import numpy as np
import logging
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from src.sentiment.vader_analyzer import VADERSentimentAnalyzer
from src.sentiment.finbert_analyzer import FinBERTSentimentAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_articles(filepath: str) -> pd.DataFrame:
    """Load articles from CSV file"""
    logger.info(f"Loading articles from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} articles")
    return df


def compute_vader_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VADER sentiment scores for articles.
    
    Returns compound score normalized to [-1, +1] scale.
    """
    logger.info("Computing VADER sentiment scores...")
    analyzer = VADERSentimentAnalyzer()
    
    vader_scores = []
    for idx, row in df.iterrows():
        text = row['clean_text']
        result = analyzer.analyze(text)
        vader_scores.append({
            'vader_positive': result['positive'],
            'vader_negative': result['negative'],
            'vader_neutral': result['neutral'],
            'vader_compound': result['compound'],  # This is already -1 to +1
            'vader_label': result['label'],
            'vader_confidence': result['confidence']
        })
        
        if (idx + 1) % 5 == 0:
            logger.info(f"  Scored {idx + 1}/{len(df)} articles with VADER")
    
    vader_df = pd.DataFrame(vader_scores)
    logger.info("✓ VADER scoring complete")
    return vader_df


def compute_finbert_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute FinBERT sentiment scores for articles.
    
    Returns normalized sentiment score:
    - -1.0 for negative
    - 0.0 for neutral
    - +1.0 for positive
    """
    logger.info("Computing FinBERT sentiment scores...")
    analyzer = FinBERTSentimentAnalyzer()
    
    finbert_scores = []
    for idx, row in df.iterrows():
        text = row['clean_text']
        result = analyzer.analyze(text)
        
        # Normalize probability to use max confidence as well
        # score_map gives us -1, 0, or 1; multiply by confidence for magnitude
        normalized_score = result['score'] * result['confidence']
        
        finbert_scores.append({
            'finbert_label': result['label'],
            'finbert_score': result['score'],  # -1, 0, or +1
            'finbert_normalized': normalized_score,  # Score * confidence
            'finbert_confidence': result['confidence'],
            'finbert_negative': result['all_scores'].get('negative', 0),
            'finbert_neutral': result['all_scores'].get('neutral', 0),
            'finbert_positive': result['all_scores'].get('positive', 0)
        })
        
        if (idx + 1) % 5 == 0:
            logger.info(f"  Scored {idx + 1}/{len(df)} articles with FinBERT")
    
    finbert_df = pd.DataFrame(finbert_scores)
    logger.info("✓ FinBERT scoring complete")
    return finbert_df


def compute_correlations(df: pd.DataFrame) -> dict:
    """
    Compute correlation metrics between VADER and FinBERT.
    
    Correlations computed on:
    1. Compound score (VADER) vs Normalized score (FinBERT)
    2. Confidence scores
    3. Raw probability distributions
    """
    logger.info("\n" + "="*70)
    logger.info("CORRELATION ANALYSIS: VADER vs FinBERT")
    logger.info("="*70)
    
    correlations = {}
    
    # 1. Main sentiment scores
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
    
    logger.info("\n1. PRIMARY SCORES (Compound/Normalized)")
    logger.info(f"   VADER compound vs FinBERT normalized:")
    logger.info(f"     Pearson r:  {pearson_r:.4f} (p={pearson_p:.4f})")
    logger.info(f"     Spearman r: {spearman_r:.4f} (p={spearman_p:.4f})")
    logger.info(f"     Kendall τ:  {kendall_tau:.4f} (p={kendall_p:.4f})")
    
    # 2. Confidence scores
    vader_conf = df['vader_confidence'].values
    finbert_conf = df['finbert_confidence'].values
    
    conf_pearson, conf_pearson_p = pearsonr(vader_conf, finbert_conf)
    conf_spearman, conf_spearman_p = spearmanr(vader_conf, finbert_conf)
    
    correlations['confidence_scores'] = {
        'pearson_r': conf_pearson,
        'pearson_p': conf_pearson_p,
        'spearman_r': conf_spearman,
        'spearman_p': conf_spearman_p
    }
    
    logger.info("\n2. CONFIDENCE SCORES")
    logger.info(f"   VADER confidence vs FinBERT confidence:")
    logger.info(f"     Pearson r:  {conf_pearson:.4f} (p={conf_pearson_p:.4f})")
    logger.info(f"     Spearman r: {conf_spearman:.4f} (p={conf_spearman_p:.4f})")
    
    # 3. Label agreement
    vader_labels = df['vader_label'].values
    finbert_labels = df['finbert_label'].values
    
    agreement = (vader_labels == finbert_labels).sum() / len(df)
    
    correlations['label_agreement'] = agreement
    logger.info(f"\n3. LABEL AGREEMENT")
    logger.info(f"   Exact match rate: {agreement:.1%} ({(vader_labels == finbert_labels).sum()}/{len(df)})")
    
    # Label mapping for comparison
    label_map = {'negative': -1, 'neutral': 0, 'positive': 1}
    vader_numeric = np.array([label_map[l] for l in vader_labels])
    finbert_numeric = np.array([label_map[l] for l in finbert_labels])
    
    label_pearson, label_pearson_p = pearsonr(vader_numeric, finbert_numeric)
    label_spearman, label_spearman_p = spearmanr(vader_numeric, finbert_numeric)
    
    correlations['label_numeric'] = {
        'pearson_r': label_pearson,
        'pearson_p': label_pearson_p,
        'spearman_r': label_spearman,
        'spearman_p': label_spearman_p
    }
    
    logger.info(f"   Label numeric correlation (numeric encoding):")
    logger.info(f"     Pearson r:  {label_pearson:.4f} (p={label_pearson_p:.4f})")
    logger.info(f"     Spearman r: {label_spearman:.4f} (p={label_spearman_p:.4f})")
    
    # 4. Direction consistency (both positive, both negative, or both neutral)
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
    
    logger.info(f"\n4. DIRECTION AGREEMENT")
    logger.info(f"   Both positive: {both_positive}/{len(df)}")
    logger.info(f"   Both negative: {both_negative}/{len(df)}")
    logger.info(f"   Both neutral:  {both_neutral}/{len(df)}")
    logger.info(f"   Total agreement: {direction_agreement:.1%}")
    
    return correlations


def print_detailed_comparison(df: pd.DataFrame):
    """Print side-by-side comparison of sentiment scores"""
    logger.info("\n" + "="*70)
    logger.info("DETAILED ARTICLE-BY-ARTICLE COMPARISON")
    logger.info("="*70)
    
    for idx, row in df.iterrows():
        logger.info(f"\n{idx + 1}. {row['firm_name']} - {row['title'][:50]}...")
        logger.info(f"   Text: {row['clean_text'][:80]}...")
        logger.info(f"   ")
        logger.info(f"   VADER Score:  {row['vader_compound']:+.4f} ({row['vader_label']:8s}) confidence={row['vader_confidence']:.3f}")
        logger.info(f"   FinBERT Score: {row['finbert_normalized']:+.4f} ({row['finbert_label']:8s}) confidence={row['finbert_confidence']:.3f}")
        logger.info(f"   ")
        logger.info(f"   VADER probs:    pos={row['vader_positive']:.3f} neu={row['vader_neutral']:.3f} neg={row['vader_negative']:.3f}")
        logger.info(f"   FinBERT probs:  pos={row['finbert_positive']:.3f} neu={row['finbert_neutral']:.3f} neg={row['finbert_negative']:.3f}")


def create_visualizations(df: pd.DataFrame):
    """Create comparison visualizations"""
    logger.info("\nCreating visualizations...")
    
    # Figure with 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VADER vs FinBERT Sentiment Comparison', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Sentiment scores
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['vader_compound'], df['finbert_normalized'], 
                         s=100, alpha=0.6, c=df.index, cmap='viridis', edgecolors='black')
    ax1.plot([-1, 1], [-1, 1], 'r--', alpha=0.5, label='Perfect Agreement')
    ax1.set_xlabel('VADER Compound Score', fontsize=11)
    ax1.set_ylabel('FinBERT Normalized Score', fontsize=11)
    ax1.set_title('Sentiment Score Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    
    # 2. Confidence score comparison
    ax2 = axes[0, 1]
    ax2.scatter(df['vader_confidence'], df['finbert_confidence'], 
               s=100, alpha=0.6, c=df.index, cmap='viridis', edgecolors='black')
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Agreement')
    ax2.set_xlabel('VADER Confidence', fontsize=11)
    ax2.set_ylabel('FinBERT Confidence', fontsize=11)
    ax2.set_title('Confidence Score Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
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
    ax3.bar(x - width/2, vader_counts, width, label='VADER', alpha=0.8)
    ax3.bar(x + width/2, finbert_counts, width, label='FinBERT', alpha=0.8)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Sentiment Label Distribution')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Residuals (difference between scores)
    ax4 = axes[1, 1]
    residuals = df['vader_compound'] - df['finbert_normalized']
    ax4.scatter(df['vader_compound'], residuals, s=100, alpha=0.6, 
               c=df.index, cmap='viridis', edgecolors='black')
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Difference')
    ax4.set_xlabel('VADER Compound Score', fontsize=11)
    ax4.set_ylabel('Residual (VADER - FinBERT)', fontsize=11)
    ax4.set_title('Score Residuals')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    output_path = 'outputs/vader_finbert_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Visualization saved to {output_path}")
    plt.close()


def main():
    """Main execution"""
    logger.info("\n" + "="*70)
    logger.info("VADER vs FinBERT SENTIMENT COMPARISON")
    logger.info("="*70)
    
    # Load articles
    df = load_articles('data/processed/glp1_relevant.csv')
    
    # Compute VADER scores
    vader_df = compute_vader_scores(df)
    df = pd.concat([df, vader_df], axis=1)
    
    # Compute FinBERT scores
    finbert_df = compute_finbert_scores(df)
    df = pd.concat([df, finbert_df], axis=1)
    
    # Compute correlations
    correlations = compute_correlations(df)
    
    # Print detailed comparison
    print_detailed_comparison(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Save combined results
    output_file = 'outputs/vader_finbert_comparison.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Combined results saved to {output_file}")
    
    # Save correlation summary
    summary_file = 'outputs/correlation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("VADER vs FinBERT Correlation Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Sample size: {len(df)} articles\n\n")
        
        f.write("PRIMARY SCORES (Compound/Normalized):\n")
        c = correlations['compound_scores']
        f.write(f"  Pearson r:  {c['pearson_r']:.4f} (p={c['pearson_p']:.4f})\n")
        f.write(f"  Spearman r: {c['spearman_r']:.4f} (p={c['spearman_p']:.4f})\n")
        f.write(f"  Kendall τ:  {c['kendall_tau']:.4f} (p={c['kendall_p']:.4f})\n\n")
        
        f.write("CONFIDENCE SCORES:\n")
        c = correlations['confidence_scores']
        f.write(f"  Pearson r:  {c['pearson_r']:.4f} (p={c['pearson_p']:.4f})\n")
        f.write(f"  Spearman r: {c['spearman_r']:.4f} (p={c['spearman_p']:.4f})\n\n")
        
        f.write("LABEL AGREEMENT:\n")
        f.write(f"  Exact match: {correlations['label_agreement']:.1%}\n\n")
        
        f.write("DIRECTION AGREEMENT:\n")
        d = correlations['direction_agreement']
        f.write(f"  Both positive: {d['both_positive']}/{len(df)}\n")
        f.write(f"  Both negative: {d['both_negative']}/{len(df)}\n")
        f.write(f"  Both neutral:  {d['both_neutral']}/{len(df)}\n")
        f.write(f"  Total: {d['overall']:.1%}\n\n")
    
    logger.info(f"✓ Correlation summary saved to {summary_file}")
    
    logger.info("\n" + "="*70)
    logger.info("✓ ANALYSIS COMPLETE")
    logger.info("="*70)
    
    return df, correlations


if __name__ == '__main__':
    df, correlations = main()
