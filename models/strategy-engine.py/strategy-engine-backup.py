
from pprint import pprint


# ==========================================
# 1. ACCOUNT HEALTH ANALYSIS
# ==========================================

def calculate_account_health(metrics):

    engagement_rate = metrics.get("avg_engagement_rate", 0)
    posting_frequency = metrics.get("posts_per_week", 0)

    findings = []

    # Engagement analysis
    if engagement_rate < 2:
        findings.append({
            "category": "engagement",
            "status": "needs_attention",
            "finding": "Engagement rate is relatively low."
        })
    else:
        findings.append({
            "category": "engagement",
            "status": "healthy",
            "finding": "Engagement rate is above the initial threshold."
        })

    # Consistency analysis
    if posting_frequency < 3:
        findings.append({
            "category": "consistency",
            "status": "needs_attention",
            "finding": "Posting frequency is relatively low."
        })
    else:
        findings.append({
            "category": "consistency",
            "status": "healthy",
            "finding": "Posting frequency meets the initial threshold."
        })

    return findings


# ==========================================
# 2. CONTENT PERFORMANCE ANALYSIS
# ==========================================

def analyse_content_performance(metrics):

    reels = metrics.get("share_reels", 0)
    carousels = metrics.get("share_carousels", 0)
    images = metrics.get("share_images", 0)

    findings = []

    if reels > carousels and reels > images:

        findings.append({
            "category": "content",
            "finding": "Reels have the highest content share.",
            "recommendation": "Evaluate Reel performance and test new formats."
        })

    elif carousels > reels and carousels > images:

        findings.append({
            "category": "content",
            "finding": "Carousels have the highest content share.",
            "recommendation": "Test educational and save-worthy carousel content."
        })

    elif images > reels and images > carousels:

        findings.append({
            "category": "content",
            "finding": "Images have the highest content share.",
            "recommendation": "Experiment with more video-led content."
        })

    else:

        findings.append({
            "category": "content",
            "finding": "Content formats have a balanced or tied distribution.",
            "recommendation": "Compare performance across individual formats."
        })

    return findings


# ==========================================
# 3. GROWTH ANALYSIS
# ==========================================

def analyse_growth(metrics):

    followers = metrics.get("follower_count", 0)
    follows = metrics.get("total_follows_generated", 0)
    reach = metrics.get("total_reach", 0)

    findings = []

    if reach > 0:
        follow_conversion = (follows / reach) * 100
    else:
        follow_conversion = 0

    if follow_conversion < 1:

        findings.append({
            "category": "growth",
            "finding": "Reach is not converting strongly into follows.",
            "recommendation": "Improve profile positioning, bio and content CTAs."
        })

    else:

        findings.append({
            "category": "growth",
            "finding": "Reach is generating audience growth.",
            "recommendation": "Identify which content is generating the most follows."
        })

    findings.append({
        "category": "audience",
        "finding": f"Current follower count: {followers}",
        "recommendation": "Monitor follower growth over time."
    })

    return findings


# ==========================================
# 4. MASTER ACCOUNT STRATEGY
# ==========================================

def generate_account_strategy(metrics, account_type):

    health = calculate_account_health(metrics)

    content = analyse_content_performance(metrics)

    growth = analyse_growth(metrics)

    strategy = {
        "account_type": account_type,
        "account_health": health,
        "content_analysis": content,
        "growth_analysis": growth
    }

    return strategy


# ==========================================
# 5. STRATEGIC RECOMMENDATION GENERATOR
# ==========================================

def generate_recommendations(analysis):

    recommendations = []

    # Account health recommendations
    for item in analysis["account_health"]:

        if item["status"] == "needs_attention":

            if item["category"] == "consistency":
                action = (
                    "Review your posting schedule "
                    "and create a consistent content plan."
                )
            else:
                action = (
                    "Investigate engagement patterns "
                    "and improve content relevance."
                )

            recommendations.append({
                "priority": "high",
                "area": item["category"],
                "action": action
            })

    # Growth recommendations
    for item in analysis["growth_analysis"]:

        if item["category"] == "growth":

            recommendations.append({
                "priority": "high",
                "area": "growth",
                "action": item["recommendation"]
            })

    # Content recommendations
    for item in analysis["content_analysis"]:

        recommendations.append({
            "priority": "medium",
            "area": "content",
            "action": item["recommendation"]
        })

    return recommendations


# ==========================================
# 6. TESTING SECTION
# ==========================================

if __name__ == "__main__":

    sample_metrics = {

        "avg_engagement_rate": 3.2,
        "avg_reach_rate": 12.5,
        "posts_per_week": 2,

        "share_reels": 0.60,
        "share_carousels": 0.25,
        "share_images": 0.15,

        "follower_count": 12000,
        "total_follows_generated": 120,
        "total_reach": 15000

    }

    # Generate account strategy
    result = generate_account_strategy(
        sample_metrics,
        account_type="creator"
    )

    # Generate recommendations
    recommendations = generate_recommendations(result)

    # Display complete results
    print("\n========== CHIC ACCOUNT STRATEGY ==========")

    pprint(result)

    print("\n========== CHIC STRATEGIC RECOMMENDATIONS ==========")

    pprint(recommendations)
