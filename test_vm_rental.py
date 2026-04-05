# def maximizeRentalRevenue(vmStock, m):
#     vmStock = vmStock[:]  # don't mutate input
#     vmStock.sort(reverse=True)
#     n = len(vmStock)
#     min_val = vmStock[-1]
#     revenue = 0
#     rem = m

#     # Level the top elements down phase by phase
#     for i in range(n - 1):
#         if rem <= 0:
#             break
#         w = i + 1
#         top = vmStock[i]
#         nxt = vmStock[i + 1]
#         drop = top - nxt
#         if drop == 0:
#             continue

#         steps = w * drop
#         if rem >= steps:
#             level_sum = (top + nxt + 1) * drop // 2
#             revenue += w * level_sum + steps * min_val
#             rem -= steps
#         else:
#             full = rem // w
#             part = rem % w
#             if full > 0:
#                 lo = top - full + 1
#                 level_sum = (top + lo) * full // 2
#                 revenue += w * level_sum + full * w * min_val
#             if part > 0:
#                 revenue += part * (top - full + min_val)
#             rem = 0

#     # Last phase: all n elements at min_val, leveling down together
#     if rem > 0:
#         L = min_val
#         w = n
#         full_levels = min(rem // w, L)

#         if full_levels > 0:
#             bot_l = L - full_levels + 1
#             # Levels >= 2: each level l costs 2*w*l - (w-1) total
#             hi = L
#             lo = max(bot_l, 2)
#             if hi >= lo:
#                 cnt = hi - lo + 1
#                 s = (hi + lo) * cnt // 2
#                 revenue += 2 * w * s - (w - 1) * cnt
#                 rem -= cnt * w
#             # Level 1: all w steps cost 2 each
#             if bot_l <= 1:
#                 revenue += 2 * w
#                 rem -= w

#         # Partial level leftover
#         if rem > 0:
#             cur = L - full_levels
#             if cur >= 2:
#                 revenue += 2 * cur          # 1st step: min=max=cur
#                 rem -= 1
#                 if rem > 0:
#                     take = min(rem, w - 1)
#                     revenue += take * (2 * cur - 1)  # remaining steps: min=cur-1
#                     rem -= take
#             elif cur == 1:
#                 take = min(rem, w)
#                 revenue += take * 2
#                 rem -= take

#     return revenue

import heapq

def maximizeRentalRevenue(vmStock, m):
    if not vmStock or m <= 0:
        return 0

    # Find the absolute maximum to size our frequency array
    max_v = max(vmStock)
    freq = [0] * (max_v + 1)
    
    max_val = 0
    min_nz = float('inf')

    # Populate frequency array and find initial max_val and min_nz
    for v in vmStock:
        if v > 0:
            freq[v] += 1
            if v > max_val:
                max_val = v
            if v < min_nz:
                min_nz = v

    total_revenue = 0

    # Process each of the m customers
    for _ in range(m):
        # Safety break (though problem guarantees sum of vmStock > m)
        if max_val == 0:
            break
            
        # 1. Customer pays lowest non-zero + highest availability
        total_revenue += (max_val + min_nz)
        
        # 2. Update the frequencies because we rented the highest available VM
        freq[max_val] -= 1
        new_val = max_val - 1
        
        if new_val > 0:
            freq[new_val] += 1
            # If the decremented value drops below our current minimum non-zero, 
            # it becomes the new minimum non-zero
            if new_val < min_nz:
                min_nz = new_val
                
        # 3. Shift max_val down if there are no more VMs with this availability
        while max_val > 0 and freq[max_val] == 0:
            max_val -= 1

    return total_revenue


def brute_force(vmStock, m):
    """Simulate customer-by-customer for verification."""
    stock = vmStock[:]
    revenue = 0
    for _ in range(m):
        max_val = max(stock)
        non_zero = [s for s in stock if s > 0]
        min_nz = min(non_zero)
        cost = min_nz + max_val
        revenue += cost
        # Rent from the type with the highest availability
        idx = stock.index(max_val)
        stock[idx] -= 1
    return revenue


import random

def test_examples():
    # Example from problem description
    assert maximizeRentalRevenue([1, 2, 4], 4) == 15, "Example 1 failed"
    # Sample Case 0
    assert maximizeRentalRevenue([2, 1, 1, 3], 4) == 12, "Sample Case 0 failed"
    print("Example tests passed.")

def test_edge_cases():
    # Single VM type
    assert maximizeRentalRevenue([5], 3) == brute_force([5], 3)
    # All same stock
    assert maximizeRentalRevenue([3, 3, 3], 9) == brute_force([3, 3, 3], 9)
    # m = 1
    assert maximizeRentalRevenue([1, 2, 4], 1) == brute_force([1, 2, 4], 1)
    # Two types
    assert maximizeRentalRevenue([3, 1], 3) == brute_force([3, 1], 3)
    # Large uniform
    assert maximizeRentalRevenue([2, 2], 4) == brute_force([2, 2], 4)
    print("Edge case tests passed.")

def test_random():
    """Fuzz test against brute force."""
    for trial in range(5000):
        n = random.randint(1, 10)
        vmStock = [random.randint(1, 20) for _ in range(n)]
        max_m = sum(vmStock)
        m = random.randint(1, max_m - 1) if max_m > 1 else 1
        fast = maximizeRentalRevenue(vmStock, m)
        slow = brute_force(vmStock, m)
        assert fast == slow, (
            f"Mismatch on trial {trial}: vmStock={vmStock}, m={m}, "
            f"fast={fast}, brute={slow}"
        )
    print("5000 random tests passed.")

if __name__ == "__main__":
    test_examples()
    test_edge_cases()
    test_random()
