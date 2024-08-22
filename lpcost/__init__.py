import pandas as pd
import numpy as np
import pulp

# Load data from CSV files
demand_data = pd.read_csv('./lpcost/2_weekly_demand.csv')
sales_data = pd.read_csv('./lpcost/3_weekly_sales.csv')
transport_cost_data = pd.read_csv('./lpcost/4_transport_cost.csv')
initial_inventory_data = pd.read_csv('./lpcost/5_initial_inventory.csv')

# Convert lead time from days to weeks and round up
transport_cost_data = pd.merge(demand_data[['Supplier', 'Supplier Country']], transport_cost_data, on='Supplier Country')
transport_cost_data = transport_cost_data.drop_duplicates(subset=['Supplier', 'Mode'], keep='first')
transport_cost_data['Lead Time (Weeks)'] = np.ceil(transport_cost_data['Lead Time'] / 7)

demand_data.drop(columns=[
    'Lead Time (Weeks)', 'Mean Weekly Demand', 'Total_Sales', 'Safety Stock (SS)','SS (round unit up)',
    'SS (CBM)', 'SS (KG)','Reorder Point (RP=s)', 'RP (round unit up)'
], inplace=True)

sales_data = pd.merge(demand_data[['Product', 'CBM/Orderlines']], sales_data, on='Product')
sales_data['CBM Sales'] = sales_data['CBM/Orderlines'] * sales_data['Weekly Sales']

# Initialize the problem
prob = pulp.LpProblem('Inventory_Optimization', pulp.LpMinimize)

products = demand_data['Product'].unique()
suppliers = demand_data['Supplier'].unique()
transportation_modes = transport_cost_data['Mode'].unique()
weeks = range(1, 79)

ignored_products = np.array(['B-037', 'B-038', 'B-041', 'B-043'])
products = np.setdiff1d(products, ignored_products)


def convert_to_cbm_cost(cost, mode, product):
    kg_orderlines = demand_data.loc[demand_data['Product'] == product, 'Kg/Orderlines'].values[0]
    cbm_orderlines = demand_data.loc[demand_data['Product'] == product, 'CBM/Orderlines'].values[0]

    if mode == 'FCL20' or mode == 'FCL40':
        return (cost / kg_orderlines) * cbm_orderlines
    if mode == 'Air':
        return cost * cbm_orderlines
    return cost

pivoted_transport_df = transport_cost_data.copy()

pivoted_transport_df['id'] = pivoted_transport_df.groupby('Supplier Country').cumcount()

# pivoted_transport_df
pivoted_transport_df = pivoted_transport_df.pivot_table(index='Supplier Country', columns='id', values='Cost').reset_index()
pivoted_transport_df = pivoted_transport_df.iloc[:, :5]

pivoted_transport_df.columns = ['Supplier Country', 'FCL20 Cost', 'FCL40 Cost', 'LCL Cost', 'Air Cost']

modes  = ['FCL20', 'FCL40', 'LCL', 'Air']

merged_demand_data = pd.merge(demand_data, pivoted_transport_df, on='Supplier Country')

for mode in modes:
    column_name = f'{mode} Cost'
    if column_name in merged_demand_data.columns:
        # Use pandas apply method with a lambda function
        merged_demand_data[column_name] = merged_demand_data.apply(
            lambda row: convert_to_cbm_cost(row[column_name], mode, row['Product']),
            axis=1
        )
        
demand_data = merged_demand_data.copy()

# Parameters
p_max = 1.46
m_constant = 1e6
# Container Parameters
capacity_fcl20 = 32
capacity_fcl40 = 66
utilization_fcl20 = 0.65
utilization_fcl40 = 0.65
# Cost Parameters
handling_costs = 10
receiving_cost = 2.5
storing_cost = 4
picking_cost = 2.5

# Decision Variables
print("Set up variables")
X_FCL20 = pulp.LpVariable.dicts('X_FCL20', [(s, t) for s in suppliers for t in weeks], lowBound=0, cat='Integer')
X_FCL40 = pulp.LpVariable.dicts('X_FCL40', [(s, t) for s in suppliers for t in weeks], lowBound=0, cat='Integer')

# Volume Variables
V_FCL20 = pulp.LpVariable.dicts('V_FCL20', [(p, s, t) for p in products for s in suppliers for t in weeks], lowBound=0, cat='Continuous')
V_FCL40 = pulp.LpVariable.dicts('V_FCL40', [(p, s, t) for p in products for s in suppliers for t in weeks], lowBound=0, cat='Continuous')
V_LCL = pulp.LpVariable.dicts('V_LCL', [(p, s, t) for p in products for s in suppliers for t in weeks], lowBound=0, cat='Continuous')
V_AIR = pulp.LpVariable.dicts('V_AIR', [(p, s, t) for p in products for s in suppliers for t in weeks], lowBound=0, cat='Continuous')
V_TOTAL = pulp.LpVariable.dicts('V_TOTAL', [(p, s, t) for p in products for s in suppliers for t in weeks], lowBound=0, cat='Continuous')

# Pallet Variables
PL_FCL20 = pulp.LpVariable.dicts('PL_FCL20', [(p, s, t) for p in products for s in suppliers for t in weeks], lowBound=0, cat='Integer')
PL_FCL40 = pulp.LpVariable.dicts('PL_FCL40', [(p, s, t) for p in products for s in suppliers for t in weeks], lowBound=0, cat='Integer')
PL_LCL = pulp.LpVariable.dicts('PL_LCL', [(p, s, t) for p in products for s in suppliers for t in weeks], lowBound=0, cat='Integer')

WEEKT_PRODUCT_ORDER_PLACED = pulp.LpVariable.dicts('WEEKT_PRODUCT_ORDER_PLACED', [(p, t) for p in products for t in weeks], cat='Binary')
WEEKT_REORDER_REACHED = pulp.LpVariable.dicts('WEEKT_REORDER_REACHED', [(p, t) for p in products for t in weeks], cat='Binary')
WEEKT_ORDER_NOT_ARRIVED = pulp.LpVariable.dicts('WEEKT_ORDER_NOT_ARRIVED', [(p, t) for p in products for t in weeks], cat='Binary')
WEEKT_MODEM_ORDERED = pulp.LpVariable.dicts('WEEKT_MODEM_ORDERED', [(p, t, m) for p in products for t in weeks for m in transportation_modes], cat='Binary')
WEEKT_SUPPLIER_ORDER_PLACED = pulp.LpVariable.dicts('WEEKT_SUPPLIER_ORDER_PLACED', [(s, t) for s in suppliers for t in weeks], cat='Binary')
AUXILIARY_VARIABLE = pulp.LpVariable.dicts('AUXILIARY_VARIABLE', [(p, s, t) for p in products for s in suppliers for t in weeks], cat='Binary')
INVENTORY = pulp.LpVariable.dicts('Inventory', [(p, t) for p in products for t in weeks], lowBound=0, cat='Continuous')

# Extract transport costs and lead times by Supplier
transport_cost_fcl20 = transport_cost_data.set_index(['Supplier', 'Mode'])['Cost'].unstack().loc[:, 'FCL20']
transport_cost_fcl40 = transport_cost_data.set_index(['Supplier', 'Mode'])['Cost'].unstack().loc[:, 'FCL40']
transport_cost_lcl = transport_cost_data.set_index(['Supplier', 'Mode'])['Cost'].unstack().loc[:, 'LCL']
transport_cost_air = transport_cost_data.set_index(['Supplier', 'Mode'])['Cost'].unstack().loc[:, 'Air']


print(demand_data.isna().values.any() or demand_data.isnull().values.any())
print(transport_cost_data.isna().values.any() or transport_cost_data.isnull().values.any())
print(sales_data.isna().values.any() or sales_data.isnull().values.any())
print(initial_inventory_data.isna().values.any() or initial_inventory_data.isnull().values.any())

# Objective function: Minimize the total cost
print("Set up objective function")
objective = pulp.lpSum([
    demand_data.loc[demand_data['Product'] == p, 'FCL20 Cost'].values[0] * X_FCL20[(s, t)] +
    demand_data.loc[demand_data['Product'] == p, 'FCL40 Cost'].values[0] * X_FCL40[(s, t)] +
    demand_data.loc[demand_data['Product'] == p, 'LCL Cost'].values[0] * V_LCL[(p, s, t)] +
    demand_data.loc[demand_data['Product'] == p, 'Air Cost'].values[0] * 180 * V_AIR[(p, s, t)] +
    handling_costs * (X_FCL20[(s, t)] + X_FCL40[(s, t)]) +
    receiving_cost * (PL_FCL20[(p, s, t)] + PL_FCL40[(p, s, t)] + PL_LCL[(p, s, t)]) +
    storing_cost * INVENTORY[(p, t)] / p_max
    for p in products for s in suppliers for t in weeks
])

# Constraints
print("Set up constraints")
# a. Pallet capacity constraints
print("A")
for p in products:
    for s in suppliers:
        for t in weeks:
            prob += V_LCL[(p, s, t)] <= PL_LCL[(p, s, t)] * p_max
            prob += V_FCL20[(p, s, t)] <= PL_FCL20[(p, s, t)] * p_max
            prob += V_FCL40[(p, s, t)] <= PL_FCL40[(p, s, t)] * p_max

# b. Container capacity constraints
print("B")
for s in suppliers:
    for t in weeks:
        prob += pulp.lpSum([V_FCL20[(p, s, t)] for p in products]) <= (utilization_fcl20 * capacity_fcl20) * X_FCL20[(s, t)]
        prob += pulp.lpSum([V_FCL40[(p, s, t)] for p in products]) <= (utilization_fcl40 * capacity_fcl40) * X_FCL40[(s, t)]

    
# c. Inventory balance constraints; and
# d. Initial inventory constraints
print("C / D")
def get_or_else_zero(p, s, t, mode, constraint: dict):
    value = t - transport_cost_data.loc[
        (transport_cost_data['Supplier'] == s) & (transport_cost_data['Mode'] == mode), 'Lead Time (Weeks)'
    ].values[0]
    if value < 1 or value > 78:
        return 0
    else:
        return constraint[(p, s, value)]

for p in products:
    for t in weeks:
        if t == 1:
            prob += INVENTORY[(p, t)] >= sales_data.loc[sales_data['Product'] == p, 'CBM Sales'].values[0]
        else:
            prob += INVENTORY[(p, t)] == (
                INVENTORY[(p, t-1)] +
                pulp.lpSum([get_or_else_zero(p, s, t, 'FCL20', V_FCL20) +
                            get_or_else_zero(p, s, t, 'FCL40', V_FCL40) +
                            get_or_else_zero(p, s, t, 'LCL', V_LCL) +
                            get_or_else_zero(p, s, t, 'Air', V_AIR)
                            for s in suppliers]) -
                pulp.lpSum([sales_data.loc[(sales_data['Product'] == p) & (sales_data['Week No.'] == t-1), 'CBM Sales'].values[0]])
            )

# e. Reorder point constraints
print("E")
for p in products:
    for t in weeks:
        prob += INVENTORY[(p, t)] + m_constant * WEEKT_REORDER_REACHED[(p, t)] >= demand_data.loc[demand_data['Product'] == p, 'RP (CBM)'].values[0]


# f. Handling recent orders
print("F")
for p in products:
    for t in weeks:
        prob += m_constant * WEEKT_ORDER_NOT_ARRIVED[p, t] >= pulp.lpSum([ 
            WEEKT_MODEM_ORDERED[(p, t_prime, m)]
            for m in transportation_modes for t_prime in range(max(1, t - int(transport_cost_data.loc[transport_cost_data['Mode'] == m, 'Lead Time (Weeks)'].values[0])), t-1)
        ])


# g. Ordering constraint
print("G")
for p in products:
    for t in weeks:
        prob += WEEKT_PRODUCT_ORDER_PLACED[(p, t)] >= WEEKT_REORDER_REACHED[(p, t)] - WEEKT_ORDER_NOT_ARRIVED[(p, t)]
        prob += WEEKT_PRODUCT_ORDER_PLACED[(p, t)] <= WEEKT_REORDER_REACHED[(p, t)]
        prob += WEEKT_PRODUCT_ORDER_PLACED[(p, t)] <= 1 - WEEKT_ORDER_NOT_ARRIVED[(p, t)]

# h. Mode ordering constraints
print("H")
for p in products:
    for t in weeks:
        prob += m_constant * WEEKT_MODEM_ORDERED[(p, t, 'FCL20')] >= pulp.lpSum([V_FCL20[(p, s, t)] for s in suppliers])
        prob += m_constant * WEEKT_MODEM_ORDERED[(p, t, 'FCL40')] >= pulp.lpSum([V_FCL40[(p, s, t)] for s in suppliers])
        prob += m_constant * WEEKT_MODEM_ORDERED[(p, t, 'LCL')] >= pulp.lpSum([V_LCL[(p, s, t)] for s in suppliers])
        prob += m_constant * WEEKT_MODEM_ORDERED[(p, t, 'Air')] >= pulp.lpSum([V_AIR[(p, s, t)] for s in suppliers])

# i. Order volume constraints
print("I")
for p in products:
    for s in suppliers:
        for t in weeks:
            prob += WEEKT_SUPPLIER_ORDER_PLACED[(s, t)] >= WEEKT_PRODUCT_ORDER_PLACED[(p, t)]
            prob += V_TOTAL[(p, s, t)] == demand_data.loc[demand_data['Product'] == p, 'S (CBM)'].values[0] * WEEKT_PRODUCT_ORDER_PLACED[(p, t)] - AUXILIARY_VARIABLE[(p, s, t)]
            prob += AUXILIARY_VARIABLE[(p, s, t)] <= demand_data.loc[demand_data['Product'] == p, 'S (CBM)'].values[0] * WEEKT_PRODUCT_ORDER_PLACED[(p, t)]
            prob += AUXILIARY_VARIABLE[(p, s, t)] >= INVENTORY[(p, t)] - demand_data.loc[demand_data['Product'] == p, 'S (CBM)'].values[0] * (1 - WEEKT_PRODUCT_ORDER_PLACED[(p, t)])
            prob += AUXILIARY_VARIABLE[(p, s, t)] <= INVENTORY[(p, t)]
            prob += AUXILIARY_VARIABLE[(p, s, t)] >= 0
            
# j. Order volume consistency constraint
print("J")
for p in products:
    for s in suppliers:
        for t in weeks:
            prob += V_TOTAL[(p, s, t)] == pulp.lpSum([V_FCL20[(p, s, t)] + V_FCL40[(p, s, t)] + V_LCL[(p, s, t)] + W_AIR[(p, s, t)]])

# k. Ensure non-negativity for all continuous decision variables
print("K")
for p in products:
    for s in suppliers:
        for t in weeks:
            prob += V_FCL20[(p, s, t)] >= 0
            prob += V_FCL40[(p, s, t)] >= 0
            prob += V_LCL[(p, s, t)] >= 0
            prob += V_AIR[(p, s, t)] >= 0
            prob += V_TOTAL[(p, s, t)] >= 0
            prob += W_AIR[(p, s, t)] >= 0

for s in suppliers:
    for t in weeks:
        prob += X_FCL20[(s, t)] >= 0
        prob += X_FCL40[(s, t)] >= 0


for p in products:
    for t in weeks:
        prob += INVENTORY[(p, t)] >= 0

# Solve the problem
prob.setObjective(objective)
pulp.LpSolverDefault.msg = 1

print("Solving...")
prob.solve()

# Print the results
status = pulp.LpStatus[prob.status]
results = {v.name: v.varValue for v in prob.variables()}
total_cost = pulp.value(prob.objective)

# Display the status, total cost, and a subset of the results
print(f'Status: {status}')
print(f'Total Cost: {total_cost}')