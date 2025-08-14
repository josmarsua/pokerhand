from collections import Counter

POKER_HAND_GRADES = {
    10: "Royal Flush",
    9: "Straight Flush",
    8: "Four of a Kind",
    7: "Full House",
    6: "Flush",
    5: "Straight",
    4: "Three of a Kind",
    3: "Two Pair",
    2: "Pair",
    1: "High Card",
}

RANK_MAP = {
    **{str(n): n for n in range(2, 11)},
    "J": 11, "Q": 12, "K": 13, "A": 14
}

def parse_card(card: str):
    """
    Parse a card string into its rank and suit.
    Returns (rank: int, suit: str) from a card string like "10H" or "AC".
    """
    suit = card[-1]  # Last character is the suit
    rank = card[:-1]  # All but the last character is the rank
    return RANK_MAP[rank], suit

def is_straight(ranks):
    """
    Check if the ranks form a straight.
    A straight is a sequence of 5 consecutive ranks.
    Returns (is_straight: bool, high_rank: int) if it is a straight, otherwise (False, None).
    """
    unique_ranks = sorted(set(ranks))
    
    if len(unique_ranks) != 5:
        return False, None
    
    # Case A-2-3-4-5 (Ace can be low)
    if unique_ranks == [2, 3, 4, 5, 14]:
        return True, 5
    
    # Normal case
    if max(unique_ranks) - min(unique_ranks) == 4:
        return True, max(unique_ranks)
    
    return False, None

def find_poker_hand(hand):
    """
    Determine the poker hand type and return its score and name.
    Hand is a list of 5 strings representing cards, e.g., ["KH", "AH", "QH", "JH", "10H"].
    Returns a tuple (score: int, name: str).    
    """
    ranks, suits = zip(*[parse_card(card) for card in hand])
    rank_counts = Counter(ranks) #{rank : count}
    counts = sorted(rank_counts.values(), reverse=True) 
    
    is_flush = len(set(suits)) == 1  # All suits are the same 
    straight, straight_high = is_straight(ranks)

    if is_flush and straight:
        # Royal if 10-J-Q-K-A
        if set(ranks) == {10, 11, 12, 13, 14}:
            return 10, POKER_HAND_GRADES[10]  # Royal Flush
        return 9, POKER_HAND_GRADES[9]  # Straight Flush
    
    if 4 in counts:
        return 8, POKER_HAND_GRADES[8] # Four of a Kind
    
    if counts == [3, 2]:
        return 7, POKER_HAND_GRADES[7] # Full House
    
    if is_flush:
        return 6, POKER_HAND_GRADES[6] # Flush
    
    if straight:
        return 5, POKER_HAND_GRADES[5] # Straight
    
    if 3 in counts:
        return 4, POKER_HAND_GRADES[4] # Three of a Kind
    
    if counts == [2, 2, 1]:
        return 3, POKER_HAND_GRADES[3] # Two Pair
    
    if 2 in counts:
        return 2, POKER_HAND_GRADES[2] # Pair
    
    return 1, POKER_HAND_GRADES[1] # High Card

if __name__ == "__main__":
    tests = [
        (["KH","AH","QH","JH","10H"], "Royal Flush"),
        (["QC","JC","10C","9C","8C"], "Straight Flush"),
        (["5C","5S","5H","5D","QH"], "Four of a Kind"),
        (["2H","2D","2S","10H","10C"], "Full House"),
        (["2D","KD","7D","6D","5D"], "Flush"),
        (["JC","10H","9C","8C","7D"], "Straight"),
        (["10H","10C","10D","2D","5S"], "Three of a Kind"),
        (["KD","KH","5C","5S","6D"], "Two Pair"),
        (["2D","2S","9C","KD","10C"], "Pair"),
        (["KD","5H","2D","10C","JH"], "High Card"),
        (["AS","2S","3D","4H","5C"], "Straight"),  # A-2-3-4-5 (wheel)
    ]
    for hand, expected in tests:
        score, name = find_poker_hand(hand)
        print(hand, "=>", name, "(score:", score, ") | esperado:", expected)