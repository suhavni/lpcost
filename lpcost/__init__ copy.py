import pandas as pd
import pulp

# Load data from CSV files
demand_data = pd.read_csv('./lpcost/2_weekly_demand.csv')
sales_data = pd.read_csv('./lpcost/3_sales.csv')
transport_cost_data = pd.read_csv('./lpcost/4_transport_cost.csv')
initial_inventory_data = pd.read_csv('./lpcost/5_initial_inventory.csv')

# Initialize the problem
prob = pulp.LpProblem("Inventory_Optimization", pulp.LpMinimize)

# Define Sets
products = demand_data['Product'].unique()
suppliers = demand_data['Supplier'].unique()
supplier_countries = demand_data['Supplier Country'].unique()
weeks = range(1, 78)

products = demand_data['Product'].unique()
ignored_products = np.array(['B-037', 'B-038', 'B-041', 'B-043'])
products = np.setdiff1d(products, ignored_products)

# Decision Variables
X_FCL20 = pulp.LpVariable.dicts("X_FCL20", [(p, c, t) for p in products for c in supplier_countries for t in weeks], lowBound=0, cat='Continuous')
X_FCL40 = pulp.LpVariable.dicts("X_FCL40", [(p, c, t) for p in products for c in supplier_countries for t in weeks], lowBound=0, cat='Continuous')
V_LCL = pulp.LpVariable.dicts("V_LCL", [(p, c, t) for p in products for c in supplier_countries for t in weeks], lowBound=0, cat='Continuous')
V_AIR = pulp.LpVariable.dicts("V_AIR", [(p, c, t) for p in products for c in supplier_countries for t in weeks], lowBound=0, cat='Continuous')
Inventory = pulp.LpVariable.dicts("Inventory", [(p, t) for p in products for t in weeks], lowBound=0, cat='Continuous')
Order = pulp.LpVariable.dicts("Order", [(p, s, t) for p in products for s in suppliers for t in weeks], cat='Binary')

# Cost Parameters from the PDF
handling_costs = 10  # 10 AUD/container
receiving_cost = 2.5  # 2.5 AUD/pallet
storing_cost = 4  # 4 AUD/pallet/week
picking_cost = 2.5  # 2.5 AUD/order

# Extract transport costs and lead times by Supplier Country
transport_cost_fcl20 = transport_cost_data.set_index(['Supplier Country', 'Mode'])['Cost'].unstack().loc[:, 'FCL20']
transport_cost_fcl40 = transport_cost_data.set_index(['Supplier Country', 'Mode'])['Cost'].unstack().loc[:, 'FCL40']
transport_cost_lcl = transport_cost_data.set_index(['Supplier Country', 'Mode'])['Cost'].unstack().loc[:, 'LCL']
transport_cost_air = transport_cost_data.set_index(['Supplier Country', 'Mode'])['Cost'].unstack().loc[:, 'Air']

# Define a large constant M (Big-M method)
M = 1e6  # Adjust M based on your problem's scale

# Objective function: Minimize the total cost
prob += pulp.lpSum([
    transport_cost_fcl20[c] * X_FCL20[(p, c, t)] +
    transport_cost_fcl40[c] * X_FCL40[(p, c, t)] +
    transport_cost_lcl[c] * V_LCL[(p, c, t)] +
    transport_cost_air[c] * V_AIR[(p, c, t)] +
    handling_costs * (X_FCL20[(p, c, t)] + X_FCL40[(p, c, t)]) +
    receiving_cost * Order[(p, s, t)] +
    storing_cost * Inventory[(p, t)]
    for p in products for c in supplier_countries for s in suppliers for t in weeks
])

# Constraints

# Inventory balance constraints
for p in products:
    for t in weeks:
        if t == 1:
            prob += Inventory[(p, t)] == initial_inventory_data.loc[initial_inventory_data['Product'] == p, 'First lot ordered'].values[0]
        else:
            prob += Inventory[(p, t)] == (
                Inventory[(p, t-1)] +
                pulp.lpSum([X_FCL20[(p, demand_data.loc[demand_data['Supplier'] == s, 'Supplier Country'].values[0], t)] +
                            X_FCL40[(p, demand_data.loc[demand_data['Supplier'] == s, 'Supplier Country'].values[0], t)] +
                            V_LCL[(p, demand_data.loc[demand_data['Supplier'] == s, 'Supplier Country'].values[0], t)] +
                            V_AIR[(p, demand_data.loc[demand_data['Supplier'] == s, 'Supplier Country'].values[0], t)]
                            for s in suppliers]) -
                demand_data.loc[demand_data['Product'] == p, 'Mean Weekly Demand'].values[0]
            )

# Reorder point constraints
for p in products:
    for s in suppliers:
        for t in weeks:
            prob += Inventory[(p, t)] >= demand_data.loc[demand_data['Product'] == p, 'Reorder Point (RP=s)'].values[0] * Order[(p, s, t)]

# No duplicate orders within lead time
for c in supplier_countries:
    for t in weeks:
        lead_time_fcl20 = transport_cost_data.loc[(transport_cost_data['Supplier Country'] == c) & (transport_cost_data['Mode'] == 'FCL20'), 'Lead Time'].values[0]
        lead_time_lcl = transport_cost_data.loc[(transport_cost_data['Supplier Country'] == c) & (transport_cost_data['Mode'] == 'LCL'), 'Lead Time'].values[0]
        lead_time_air = transport_cost_data.loc[(transport_cost_data['Supplier Country'] == c) & (transport_cost_data['Mode'] == 'Air'), 'Lead Time'].values[0]
        
        if t + lead_time_fcl20 <= max(weeks):
            prob += pulp.lpSum([Order[(p, s, t_prime)] for s in suppliers for t_prime in range(t, int(t + lead_time_fcl20)) if demand_data.loc[demand_data['Supplier'] == s, 'Supplier Country'].values[0] == c]) <= 1

        if t + lead_time_lcl <= max(weeks):
            prob += pulp.lpSum([Order[(p, s, t_prime)] for s in suppliers for t_prime in range(t, int(t + lead_time_lcl)) if demand_data.loc[demand_data['Supplier'] == s, 'Supplier Country'].values[0] == c]) <= 1

        if t + lead_time_air <= max(weeks):
            prob += pulp.lpSum([Order[(p, s, t_prime)] for s in suppliers for t_prime in range(t, int(t + lead_time_air)) if demand_data.loc[demand_data['Supplier'] == s, 'Supplier Country'].values[0] == c]) <= 1

# Modify the constraint using the Big-M method
for p in products:
    for s in suppliers:
        c = demand_data.loc[demand_data['Supplier'] == s, 'Supplier Country'].values[0]
        for t in weeks:
            prob += X_FCL20[(p, c, t)] >= (demand_data.loc[demand_data['Product'] == p, 'Order-Up-To Level (S)'].values[0] - Inventory[(p, t)]) - M * (1 - Order[(p, s, t)])

# Solve the problem
prob.solve()

# Print the results
status = pulp.LpStatus[prob.status]
results = {v.name: v.varValue for v in prob.variables()}
total_cost = pulp.value(prob.objective)

print(f"Status: {status}")
print(f"Total Cost: {total_cost}")

with open("results.csv", "w") as f:
    f.write("Variable,Value")
    for k, v in results.items():
        if v < 0:
            print(f"{k}: {v}")
        else:
            f.write(f"{k},{v}")

# # Display the status, total cost, and a subset of the results
# print(f"Status: {status}")
# print(f"Total Cost: {total_cost}")
# for k, v in results.items():
#     if v > 0:
#         print(f"{k}: {v}")