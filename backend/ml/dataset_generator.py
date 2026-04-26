import pandas as pd
import numpy as np
import os

def generate_supplier_data(num_records=2500):
    np.random.seed(42)
    
    # 1. Base Features
    supplier_id = [f"SUP-{str(i).zfill(5)}" for i in range(num_records)]
    
    # Privileged Demographics distribution
    # business_size: 1 (Large), 0 (Small/Micro) - 30% Large, 70% Small
    business_size = np.random.choice([1, 0], size=num_records, p=[0.3, 0.7])
    
    # location_tier: 1 (Tier-1), 0 (Tier-3/Flood-prone) - 40% Tier-1, 60% Tier-3
    location_tier = np.random.choice([1, 0], size=num_records, p=[0.4, 0.6])
    
    # owner_gender: 1 (Male), 0 (Women/Minority) - 75% Male, 25% Women/Minority
    owner_gender = np.random.choice([1, 0], size=num_records, p=[0.75, 0.25])
    
    # 2. Actual Operational Reliability KPIs
    # years_active: 1 to 20 years
    years_active = np.random.randint(1, 21, size=num_records)
    
    # on_time_delivery_rate: Base delivery rate between 70% and 99%
    # Adding a slight natural correlation with business size and years active
    base_delivery_rate = np.random.uniform(70, 99, size=num_records) + (business_size * 2) + (years_active * 0.1)
    on_time_delivery_rate = np.clip(base_delivery_rate, 70, 100)
    
    # defect_rate: Percentage of defective items (0.5% to 5%)
    defect_rate = np.random.uniform(0.5, 5.0, size=num_records) - (business_size * 0.5)
    defect_rate = np.clip(defect_rate, 0.1, 10.0)
    
    # 3. Calculate True Performance Score (Hidden)
    # A perfectly fair metric based purely on KPIs
    true_performance_score = (on_time_delivery_rate * 0.7) + ((10 - defect_rate) * 3)
    # Normalize to 0-100 range
    true_performance_score = (true_performance_score - true_performance_score.min()) / (true_performance_score.max() - true_performance_score.min()) * 100
    
    # 4. Inject Systematic Bias for AI Trust Score
    # We introduce a 'Region Penalty' for Tier-3 cities (location_tier = 0)
    # The penalty is masked by adding noise and interacting with business_size
    
    # Systematic bias: -8 points for Tier-3, but less penalty if they are a large business
    region_penalty = np.where((location_tier == 0) & (business_size == 0), -12, 
                              np.where(location_tier == 0, -4, 0))
    
    # Add gender bias penalty: -5 points for women/minority owners
    gender_penalty = np.where(owner_gender == 0, -5, 0)
    
    # Assigned AI Trust Score
    ai_trust_score = true_performance_score + region_penalty + gender_penalty + np.random.normal(0, 2, num_records)
    ai_trust_score = np.clip(ai_trust_score, 0, 100)
    
    # 5. Build DataFrame
    df = pd.DataFrame({
        'supplier_id': supplier_id,
        'business_size': business_size,
        'location_tier': location_tier,
        'owner_gender': owner_gender,
        'years_active': years_active,
        'on_time_delivery_rate': np.round(on_time_delivery_rate, 2),
        'defect_rate': np.round(defect_rate, 2),
        'true_performance_score': np.round(true_performance_score, 2),
        'ai_trust_score': np.round(ai_trust_score, 2)
    })
    
    # 6. Contract Awarded (Target Variable)
    # Threshold for receiving a contract is an AI score >= 75
    df['contract_awarded'] = (df['ai_trust_score'] >= 75).astype(int)
    
    # Save to CSV
    os.makedirs(os.path.dirname('c:/Users/hp/Desktop/Nirmayee/Projects/FairChain/backend/data/'), exist_ok=True)
    file_path = 'c:/Users/hp/Desktop/Nirmayee/Projects/FairChain/backend/data/supplier_database.csv'
    df.to_csv(file_path, index=False)
    
    print(f"Generated dataset with {num_records} records.")
    print(f"Saved to: {file_path}")
    
    # Quick sanity check on bias
    print("\n--- Bias Sanity Check (Contract Award Rate) ---")
    print("Privileged vs Unprivileged based on Location Tier:")
    print(df.groupby('location_tier')['contract_awarded'].mean())
    
if __name__ == "__main__":
    generate_supplier_data()
