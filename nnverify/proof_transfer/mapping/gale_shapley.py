from typing import List, Tuple, Dict

def _interval_score(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """S(I,J)=|I∩J| / (|I|·|J|)  (Eq. 10 in the proposal)."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    return inter / ((a[1]-a[0]) * (b[1]-b[0]) + 1e-12)

def layer_match(ranges_A: List[Tuple[float, float]],
                ranges_B: List[Tuple[float, float]]) -> Dict[int, int]:
    """
    Returns {b-idx → a-idx} so every neuron in the *right* (updated net)
    is matched to at least one neuron on the *left* (original net).
    """
    left_nodes  = list(range(len(ranges_A)))
    right_nodes = list(range(len(ranges_B)))

    # Build weight dictionary
    edge_weights = {
        (j, i): _interval_score(ranges_B[j], ranges_A[i])
        for j in right_nodes for i in left_nodes
    }

    # First pass
    right_to_left, _, _ = gale_shapley_weighted_complete(
        left_nodes, right_nodes, edge_weights
    )

    # If |B| > |A|, iterate as in Section 4.2
    unmatched = [j for j, v in right_to_left.items() if v is None]
    while unmatched:
        sub_w = { (j, i): edge_weights[(j, i)]
                  for j in unmatched for i in left_nodes }
        sub_match, _, _ = gale_shapley_weighted_complete(
            left_nodes, unmatched, sub_w
        )
        right_to_left.update(sub_match)
        unmatched = [j for j, v in right_to_left.items() if v is None]

    return {j: i for j, i in right_to_left.items() if i is not None}

def gale_shapley_weighted_complete(left_nodes, right_nodes, edge_weights):
    """
    Gale-Shapley algorithm for a complete bipartite graph with weighted edges.
    All right-side nodes must be matched.
    Higher weight means higher preference.
    
    Args:
        left_nodes: List of nodes on the left side
        right_nodes: List of nodes on the right side
        edge_weights: Dict mapping (right_node, left_node) pairs to their edge weight
    """
    # edge_weights is a dictionary of edge weights; the weights are scores computed by the procedure in the slides

    m = len(left_nodes)
    n = len(right_nodes)
    
    # Calculate capacities for left nodes
    if m >= n:
        left_capacities = {left: 1 for left in left_nodes}
    else:
        # Distribute capacity evenly
        base_capacity = n // m
        extra = n % m
        left_capacities = {}
        for i, left in enumerate(left_nodes):
            left_capacities[left] = base_capacity + (1 if i < extra else 0)
    
    # Create preference lists for right nodes based on weights
    right_prefs = {}
    for right in right_nodes:
        # Sort left nodes by weight (higher = better)
        right_prefs[right] = sorted(
            left_nodes,
            key=lambda left: edge_weights.get((right, left), 0),
            reverse=True
        )
    
    # Create preference rankings for left nodes based on weights
    left_ranks = {}
    for left in left_nodes:
        # Lower rank value = higher preference
        # We use negative weight so that higher weights get lower rank values
        left_ranks[left] = {
            right: -edge_weights.get((right, left), 0) for right in right_nodes
        }
    
    # Initialize matching structures
    unmatched_right = list(right_nodes)
    left_matches = {left: [] for left in left_nodes}
    right_matches = {right: None for right in right_nodes}
    next_proposal_idx = {right: 0 for right in right_nodes}
    
    # Main algorithm loop
    while unmatched_right:
        right = unmatched_right.pop(0)
        
        # Check if right node has exhausted preferences
        if next_proposal_idx[right] >= len(right_prefs[right]):
            print(f"Warning: {right} has exhausted all preferences and remains unmatched.")
            continue
        
        # Get right node's next preferred left node
        left = right_prefs[right][next_proposal_idx[right]]
        next_proposal_idx[right] += 1
        
        # Case 1: Left node has capacity available
        if len(left_matches[left]) < left_capacities[left]:
            left_matches[left].append(right)
            right_matches[right] = left
        
        # Case 2: Left node must compare preferences
        else:
            # Consider all candidates including the new proposal
            candidates = left_matches[left] + [right]
            
            # Sort by left node's preferences (using weights)
            candidates.sort(key=lambda r: left_ranks[left].get(r, float('inf')))
            
            # Keep the most preferred candidates up to capacity
            accepted = candidates[:left_capacities[left]]
            rejected = candidates[left_capacities[left]:]
            
            # Update matches
            left_matches[left] = accepted
            
            if right in accepted:
                right_matches[right] = left
            else:
                # Try again with next preference
                unmatched_right.append(right)
            
            # Handle rejections of previously matched nodes
            for rejected_right in rejected:
                if rejected_right != right and right_matches[rejected_right] == left:
                    right_matches[rejected_right] = None
                    unmatched_right.append(rejected_right)
    
    # Calculate total weight of the matching
    total_weight = sum(
        edge_weights.get((right, left), 0)
        for right, left in right_matches.items()
        if left is not None
    )
    
    return right_matches, left_matches, total_weight
