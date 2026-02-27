# Corrected main.py

# Import necessary modules
from inventory_risk_check import InventoryRiskChecker

# Initialize the risk checker
risk_checker = InventoryRiskChecker()

# Main function to run risk assessment
if __name__ == '__main__':
    risk_check_result = risk_checker.check_inventory_risk()
    print(risk_check_result)