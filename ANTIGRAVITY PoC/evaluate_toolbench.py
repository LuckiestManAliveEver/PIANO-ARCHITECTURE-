import json
from typing import List, Dict

# Mocking the Piano Components locally to allow "exact one correct key" logic 
# and slightly broader keyword detection without modifying the core files too much.

class EnhancedIntentMatrix:
    def __init__(self):
        # Rule-based routing for simulation
        self.rules = {
            "K1": ["calculate", "profit", "math", "sum", "add", "multiply", "divide", "solve", "equation", "compute"],
            "K2": ["verify", "check", "validate", "confirm", "audit", "review", "proofread", "test", "ensure"],
            "K3": ["format", "summarize", "style", "json", "markdown", "prettify", "report", "convert", "table"]
        }

    def decode_intents(self, query: str) -> List[str]:
        query_lower = query.lower()
        detected = []
        
        # Priority check or multi-label? 
        # User said "Each query maps to exactly one correct Key".
        # But Conductor might select multiple.
        # We will detect all matches.
        
        for key, keywords in self.rules.items():
            if any(k in query_lower for k in keywords):
                detected.append(key)
        
        # Fallback or tie-breaking if empty?
        if not detected:
            # Random or default? Let's say none.
            pass
            
        return detected

def generate_dataset():
    # 50 Queries with 1 Gold Key each
    dataset = [
        # K1 - Math
        {"q": "Calculate the profit.", "gold": "K1"},
        {"q": "Solve this equation: 2+2", "gold": "K1"},
        {"q": "Compute the total cost.", "gold": "K1"},
        {"q": "What is the sum of these numbers?", "gold": "K1"},
        {"q": "Multiply 5 by 10.", "gold": "K1"},
        {"q": "Divide the revenue by 2.", "gold": "K1"},
        {"q": "Add these items up.", "gold": "K1"},
        {"q": "Math problem: x^2 = 4", "gold": "K1"},
        {"q": "Calculate the area.", "gold": "K1"},
        {"q": "Compute the derivative.", "gold": "K1"},
        {"q": "What is 15 percent of 100?", "gold": "K1"},
        {"q": "Solve for x.", "gold": "K1"},
        {"q": "How many apples do I need?", "gold": "K1"},
        {"q": "Calculate interest rate.", "gold": "K1"},
        {"q": "Compute velocity.", "gold": "K1"},
        {"q": "Solve the integral.", "gold": "K1"},
        {"q": "Add 50 to the account.", "gold": "K1"},
        {"q": "Multiply the matrix.", "gold": "K1"},
        {"q": "Calculate the mean.", "gold": "K1"},
        {"q": "Compute the variance.", "gold": "K1"},

        # K2 - Verify
        {"q": "Verify the results.", "gold": "K2"},
        {"q": "Check if this is correct.", "gold": "K2"},
        {"q": "Validate the inputs.", "gold": "K2"},
        {"q": "Confirm the transaction.", "gold": "K2"},
        {"q": "Audit the logs.", "gold": "K2"},
        {"q": "Review the code.", "gold": "K2"},
        {"q": "Proofread this text.", "gold": "K2"},
        {"q": "Test the connection.", "gold": "K2"},
        {"q": "Ensure compliance.", "gold": "K2"},
        {"q": "Double check the math.", "gold": "K2"}, # Tricky: contains 'math' but intent is check
        {"q": "Verify the checksum.", "gold": "K2"},
        {"q": "Validate the user.", "gold": "K2"},
        {"q": "Check for errors.", "gold": "K2"},
        {"q": "Confirm the date.", "gold": "K2"},
        {"q": "Audit this file.", "gold": "K2"},
        {"q": "Review the summary.", "gold": "K2"},
        {"q": "Test the function.", "gold": "K2"},

        # K3 - Format
        {"q": "Format this as JSON.", "gold": "K3"},
        {"q": "Summarize the findings.", "gold": "K3"},
        {"q": "Convert to Markdown.", "gold": "K3"},
        {"q": "Make a table.", "gold": "K3"},
        {"q": "Prettify this output.", "gold": "K3"},
        {"q": "Generate a report.", "gold": "K3"},
        {"q": "Style this text.", "gold": "K3"},
        {"q": "Format in CSV.", "gold": "K3"},
        {"q": "Convert to PDF.", "gold": "K3"},
        {"q": "Summarize briefly.", "gold": "K3"},
        {"q": "Make it look good.", "gold": "K3"}, # 'look' not in keywords? Might miss.
        {"q": "Format the date.", "gold": "K3"},
        {"q": "Report the status.", "gold": "K3"},
    ]
    
    # Fill to 50
    while len(dataset) < 50:
        dataset.append({"q": "Calculate something else.", "gold": "K1"})
        
    return dataset

def evaluate():
    dataset = generate_dataset()
    router = EnhancedIntentMatrix()
    
    y_true = []
    y_pred = [] # Primary prediction (if multiple, we assume the system failed or we take the first)
    
    # Routing stats
    correct_count = 0
    false_activations = 0 # Extra keys activated
    missed_activations = 0 # Correct key NOT activated
    
    print(f"Running ToolBench Evaluation on {len(dataset)} queries...\n")
    
    for item in dataset:
        query = item['q']
        gold = item['gold']
        
        predicted_keys = router.decode_intents(query)
        
        y_true.append(gold)
        
        # Metric Logic
        hit = False
        if gold in predicted_keys:
            hit = True
            
        # Select primary prediction for Confusion Matrix
        # If multiple, prioritize K1 > K2 > K3 or just take first.
        # If empty, use "None".
        primary_pred = predicted_keys[0] if predicted_keys else "None"
        y_pred.append(primary_pred)
        
        if predicted_keys == [gold]:
            correct_count += 1
        
        if gold not in predicted_keys:
            missed_activations += 1
            
        # False Positives: Any key in predicted that is NOT gold
        extras = [k for k in predicted_keys if k != gold]
        false_activations += len(extras)

    accuracy = (correct_count / len(dataset)) * 100
    missed_rate = (missed_activations / len(dataset)) * 100
    # False Activation Rate: % of queries having false activations.
    queries_with_extras = 0
    for i, item in enumerate(dataset):
        gold = dataset[i]['gold'] # Access from dataset directly
        pred = router.decode_intents(dataset[i]['q'])
        if any(k != gold for k in pred):
            queries_with_extras += 1
    
    false_act_rate_pct = (queries_with_extras / len(dataset)) * 100

    print("=== Evaluation Results ===")
    print(f"Routing Accuracy (Exact Match): {accuracy:.2f}%")
    print(f"Missed Activation Rate: {missed_rate:.2f}%")
    print(f"False Activation Rate: {false_act_rate_pct:.2f}%")
    
    print("\n=== Confusion Matrix (Primary Prediction) ===")
    labels = ["K1", "K2", "K3", "None"]
    
    # Manual Confusion Matrix
    cm = {l1: {l2: 0 for l2 in labels} for l1 in labels}
    
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    
    print(f"{'True/Pred':<10} | {'K1':<5} {'K2':<5} {'K3':<5} {'None':<5}")
    print("-" * 35)
    for row_label in labels:
        # Only print rows that exist in truth (K1, K2, K3). 'None' is not a truth label.
        if row_label == "None": continue
        
        row_str = f"{row_label:<10} | "
        for col_label in labels:
            row_str += f"{cm[row_label][col_label]:<5} "
        print(row_str)

if __name__ == "__main__":
    evaluate()
