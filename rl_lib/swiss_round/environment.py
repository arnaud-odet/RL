import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import random
import networkx as nx
from multiprocessing import Pool
import copy
from tqdm import tqdm
from rl_lib.swiss_round.utils import check_probability

# Helper function for simulations using multi-processing.
def _parallel_simulation(args) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Helper function for parallel tournament simulation.
    Must be at module level for multiprocessing to work.
    
    Parameters:
    -----------
    env : SwissRoundEnv
        A copy of the environment to run the simulation
        
    Returns:
    --------
    Optional[Tuple[np.ndarray, np.ndarray]]:
        Rankings and points arrays for all teams, or None if simulation failed
    """
    env, policy = args
    try:
        n_teams = len(env.teams)
        tournament_results = env.simulate_tournament(policy=policy)
        
        # If simulation failed and returned None
        if tournament_results is None:
            return None
            
        # Initialize arrays for this simulation
        rankings = np.zeros(n_teams)
        points = np.zeros(n_teams)
        rewards = np.zeros(n_teams)
        
        # Process tournament results
        for rank, (team_id, team_points, team_reward, _, _) in enumerate(tournament_results):
            rankings[team_id] = rank + 1  # Convert to 1-based ranking
            points[team_id] = team_points
            rewards[team_id] = team_reward
        return rankings, points, rewards
        
    except Exception as e:
        #print(f"Warning: Simulation failed with error: {str(e)}")
        return None

def _imap_unordered_bar(func, args, n_processes: int = None, total: int = None, desc: str = "Processing"):
    """
    Wrapper function for tqdm with imap_unordered for parallel processing with progress bar
    """
    with Pool(processes=n_processes) as pool:
        for _ in tqdm(pool.imap_unordered(func, args), total=total, desc=desc):
            yield _

@dataclass
class Team:
    """Class representing a team in the tournament"""
    id: int
    strength: float
    points: int = 0
    opponents_avg_points: int = 0
    is_agent: bool = False
    played_against: set = None

    def __post_init__(self):
        if self.played_against is None:
            self.played_against = set()

class SwissRoundEnv:
    """Environment for Swiss-round tournament simulation"""
    
    def __init__(
        self,
        n_teams: int,
        n_rounds: int,
        name: str, 
        team_strengths: List[float],
        threshold_ranks: List[int],
        bonus_points: List[int],
        agent_id: Optional[int] = None,
        max_draw_probability:float=0.3
    ):
        """
        Initialize the Swiss-round tournament environment
        
        Args:
            n_teams: Number of teams in the tournament
            n_rounds: Number of rounds to play
            team_strengths: List of team strengths
            threshold_ranks: List of rank thresholds for bonus points
            bonus_points: List of bonus points corresponding to thresholds
            agent_id: ID of the agent team (None if no agent)
        """
        assert n_teams % 2 == 0, "Number of teams must be even"
        assert len(team_strengths) == n_teams, "Must provide strength for each team"
        assert len(threshold_ranks) == len(bonus_points), "Thresholds and bonus points must match"
        if agent_id is not None:
            assert 0 <= agent_id < n_teams, "Agent ID must be valid"
        
        self.n_teams = n_teams
        self.n_rounds = n_rounds
        self.name = name
        self.threshold_ranks = sorted(threshold_ranks)  # Sort in ascending order
        self.bonus_points = [x for _, x in sorted(zip(threshold_ranks, bonus_points))]
        self.max_draw_probability = max_draw_probability
        self.current_round = 0
        
        
        # Initialize teams
        self.teams = []
        for i in range(n_teams):
            self.teams.append(Team(
                id=i,
                strength=team_strengths[i],
                is_agent=(i == agent_id if agent_id is not None else False)
            ))
        
        self.agent = self.teams[agent_id] if agent_id is not None else None
        
        assert check_probability(team_strengths=team_strengths, max_draw_probability=max_draw_probability)
    
    # Tournament functions
    def _simulate_match(
        self,
        team1: Team,
        team2: Team,
        team1_action: str = 'win', #Only considers action of action 1 as only 1 agent is present and if one is involved in the game, it will be team 1
    ) -> Tuple[int, int]:
        """
        Simulate a match between two teams
        
        Args:
            team1: First team
            team2: Second team
            team1_action: Action chosen by team1 ('win', 'draw', 'lose')
            
        Returns:
            Tuple of points gained by each team
        """
        # Base probability of team1 winning
        strength_diff = team1.strength - team2.strength
        tmp_win_prob = 1 / (1 + np.exp(-strength_diff))
        tmp_loss_prob = 1 / (1 + np.exp(+strength_diff))
        
        # Draw probability inversely proportional to strength difference
        # Maximum draw probability is 0.3 when strengths are equal
        tmp_draw_prob = self.max_draw_probability * np.exp(-abs(strength_diff))

        win_prob = tmp_win_prob / (tmp_win_prob + tmp_draw_prob + tmp_loss_prob)
        draw_prob = tmp_draw_prob / (tmp_win_prob + tmp_draw_prob + tmp_loss_prob)
        loss_prob = tmp_loss_prob / (tmp_win_prob + tmp_draw_prob + tmp_loss_prob)

        assert abs(win_prob + loss_prob + draw_prob - 1) < 0.0001, 'Please check the probability computation'
        
        outcome = np.random.random()
        
        
        
        if outcome < win_prob:
            if team1_action == 'win' :
                return (3, 0)  
            if team1_action == 'draw' : 
                return (1,1)
            if team1_action == 'lose' :
                return (0, 3)    
        elif outcome < win_prob + draw_prob:
            if team1_action == 'lose' :
                return (0, 3)    
            else :
                return (1, 1)  
        else :
            return (0,3)
 
    def _get_opponent_average_points(self, team: Team) -> float:
        """
        Calculate the average points of all opponents faced by a team
        
        Args:
            team: Team to calculate opponent average for
            
        Returns:
            Average points of opponents, or -1 if no opponents yet
        """
        if not team.played_against:
            return -1
            
        opponent_points = [self.teams[opp_id].points for opp_id in team.played_against]
        return sum(opponent_points) / len(opponent_points)

    def _get_rankings(self) -> List[Team]:
        """
        Get current rankings of teams.
        Tiebreaker is based on average opponent points.
        """
        # First, calculate average opponent points for each team
        opponent_averages = {team.id: self._get_opponent_average_points(team) 
                           for team in self.teams}
        
        for team in self.teams :
            team.opponents_avg_points = opponent_averages[team.id]
        
        # Sort teams by points first, then by opponent average
        return sorted(
            self.teams,
            key=lambda x: (-x.points, -x.opponents_avg_points)
        )
    
    def _pair_teams_simple(self) -> List[Tuple[Team, Team]]:
        """
        Simple pairing method with random first round pairing.
        """
        rankings = self._get_rankings()
        unpaired = rankings.copy()
        pairs = []
        
        # Handle first round separately with random pairing
        if self.current_round == 0:
            # Create a copy and shuffle it
            first_round_teams = rankings.copy()
            random.shuffle(first_round_teams)
            
            # Pair teams sequentially from the shuffled list
            while len(first_round_teams) >= 2:
                team1 = first_round_teams.pop(0)
                team2 = first_round_teams.pop(0)
                pairs.append((team1, team2))
            
            return pairs
        
        def find_best_opponent(team: Team, candidates: List[Team]) -> Optional[Team]:
            """Find the best valid opponent from candidates list"""
            for opp in candidates:
                if (opp.id != team.id and 
                    opp.id not in team.played_against):
                    return opp
            return None
        
        while len(unpaired) >= 2:
            team1 = unpaired[0]  # Take highest ranked unpaired team
            
            # Try to find opponent in the same score group
            same_points = [t for t in unpaired[1:] 
                        if t.points == team1.points]
            opponent = find_best_opponent(team1, same_points)
            
            # If no opponent in same score group, look at adjacent score groups
            if opponent is None:
                opponent = find_best_opponent(team1, unpaired[1:])
            
            # If still no valid opponent, return None to trigger networkx method
            if opponent is None:
                return None
                
            pairs.append((team1, opponent))
            unpaired.remove(team1)
            unpaired.remove(opponent)
        
        return pairs

    def _pair_teams_networkx(self) -> List[Tuple[Team, Team]]:
        """
        Advanced pairing method using maximum weight matching.
        Adds small random perturbations to break ties.
        """
        rankings = self._get_rankings()
        n_teams = len(rankings)
        
        # Create bipartite graph
        G = nx.Graph()
        
        # Add nodes for both sides of bipartite graph
        left_nodes = [f'L{team.id}' for team in rankings]
        right_nodes = [f'R{team.id}' for team in rankings]
        G.add_nodes_from(left_nodes, bipartite=0)
        G.add_nodes_from(right_nodes, bipartite=1)
        
        # Try multiple times with different random perturbations
        noise_levels = 6
        attemps_per_level = 3
        max_attempts = noise_levels * attemps_per_level
        max_noise = 0.01
        for attempt in range(max_attempts):
            # Create pertubation matrix to be use later on to break ties
            noise = np.random.uniform(0,max_noise,size = (n_teams, n_teams))  
            max_noise = max_noise * (10 if (attempt +1)%attemps_per_level ==0 else 1) 
            adj_matrix = np.zeros((n_teams, n_teams))      
            # Add edges with weights
            for i, team1 in enumerate(rankings):
                for j, team2 in enumerate(rankings):
                    if team1.id == team2.id:
                        continue
                    
                    # Skip if they already played
                    if team2.id in team1.played_against:
                        continue
                    
                    # For first round, use team IDs to ensure deterministic pairing
                    if self.current_round == 0:
                        weight = 1000 - abs(team1.id - team2.id)
                    else:
                        # Calculate weight based on points difference
                        points_diff = abs(team1.points - team2.points)
                        opp_avg_pts_diff = abs(team1.opponents_avg_points - team2.opponents_avg_points)
                        
                        # Higher weight for same points group
                        if team1.points == team2.points:
                            weight = 1000 - opp_avg_pts_diff
                        else:
                            weight = 500 - 10 * points_diff - opp_avg_pts_diff
                        
                        # Add the noise comparing team ids to be sure both edges have the same noise value
                        weight += noise[min(team1.id, team2.id), max(team1.id, team2.id)]
                        adj_matrix[team1.id, team2.id] = weight
                        adj_matrix[team2.id, team1.id] = weight
                    
                    # Add edge
                    G.add_edge(f'L{team1.id}', f'R{team2.id}', weight=weight)
        
            try:
                matching = nx.max_weight_matching(G, maxcardinality=True)
                
                # Convert matching to pairs of teams
                pairs = []
                matched = set()
                
                for i, (node1, node2) in enumerate(matching):
                    weight_l = adj_matrix[int(node1[1:]), int(node2[1:])]
                    weight_r = adj_matrix[int(node2[1:]), int(node1[1:])]

                    start_str = f"Matching n° {i+1} : {node1} - {node2}, weight = {weight_l:.3f} and {weight_r:.3f}  | "
                    end_str = 'One of the team is already used'
                    # Only process each pair once
                    if node1[0] == 'L' and int(node1[1:]) not in matched and int(node2[1:]) not in matched:
                        team1_id = int(node1[1:])
                        team2_id = int(node2[1:])
                        pairs.append((self.teams[team1_id], self.teams[team2_id]))
                        matched.add(int(node1[1:]))
                        matched.add(int(node2[1:]))
                        end_str = 'Matching used for pairing'
                    elif node2[0] == 'L' and int(node1[1:]) not in matched and int(node2[1:]) not in matched:
                        team1_id = int(node2[1:])
                        team2_id = int(node1[1:])
                        pairs.append((self.teams[team1_id], self.teams[team2_id]))
                        matched.add(int(node2[1:]))
                        matched.add(int(node1[1:]))
                        end_str = 'Matching used for pairing'
                    if attempt == max_attempts -1:
                        #print(start_str + end_str + str(matched))
                        pass
                if len(pairs) == n_teams // 2:
                    # Restoring the rtandom state
                    return pairs
                    
            except nx.NetworkXError:
                if attempt < max_attempts - 1:
                    continue
        
        # If we get here, all attempts failed
        raise ValueError("Could not find a perfect matching after multiple attempts")

    def _pair_teams(self, verbose:bool= False) -> List[Tuple[Team, Team]]:
        """
        Hybrid pairing method that tries simple pairing first,
        then falls back to networkx if needed.
        """
        
        # Try simple pairing first
        pairs = self._pair_teams_simple()
        
        # If simple pairing fails, use networkx
        if pairs is None:
            if verbose:
                print("Simple pairing failed, using networkx matching...")
            pairs = self._pair_teams_networkx()
        
        return pairs
      
    def step(self, action: Optional[int] = 0, verbose:bool=False) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one step (round) in the environment
        
        Args:
            action: Agent's action (0: win, 1: draw, 2: lose), None if no agent
            
        Returns:
            (next_state, reward, done)
        """
        action_map = {0: 'win', 1: 'draw', 2: 'lose'}
        
        # Pair teams and play matches
        pairs = self._pair_teams(verbose = verbose)
        
        # Play all matches for this round
        for team1, team2 in pairs:
            # Update played_against sets
            team1.played_against.add(team2.id)
            team2.played_against.add(team1.id)
            game_str = f"Game : Team {team1.id} (points : {team1.points}, strength : {team1.strength:.2f}) vs Team {team2.id} (points : {team2.points}, strength : {team2.strength:.2f})"
            
            # If agent is involved, use the provided action
            if team1.is_agent:
                result = self._simulate_match(team1, team2, action_map[action])
                points1, points2 = result
            elif team2.is_agent:
                result = self._simulate_match(team2, team1, action_map[action])
                points2, points1 = result
            else:
                # Both teams try to win
                result = self._simulate_match(team1, team2, 'win')
                points1, points2 = result
            team1.points += points1
            team2.points += points2
            
            if verbose :
                if points1 == 3 :
                    result_str = f"Team {team1.id} wins"
                elif points2 == 3 :
                    result_str = f"Team {team2.id} wins"
                else :
                    result_str = 'Draw'
                print(f"{game_str} : {result_str}")
            
        self.current_round += 1
        done = self.current_round >= self.n_rounds
        
        # Calculate reward
        reward = 0
        if done and self.agent is not None:
            rankings = self._get_rankings()
            reward += self.agent.points
            agent_rank = next(i for i, team in enumerate(rankings) if team.is_agent)
            # Add bonus points for each threshold reached
            for rank_threshold, bonus in zip(self.threshold_ranks, self.bonus_points):
                if agent_rank < rank_threshold:
                    reward += bonus
            #print(f"Agent rank : {agent_rank}, points : {self.agent.points},  bonus : {reward - self.agent.points}")
        return self.get_state(), reward, done
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.current_round = 0
        for team in self.teams:
            team.points = 0
            team.played_against = set()
            
        return self.get_state()
    
    # RL agent state 
    def get_state(self) -> np.ndarray:
        """
        Get the current state for the agent
        
        Returns:
            state: Array containing for each opponent:
                  [strength, point_difference, has_played_against]
        """
        if self.agent is None:
            return np.array([])
            
        state = []
        for team in self.teams:
            if not team.is_agent:
                state.extend([
                    team.strength,
                    team.points - self.agent.points,
                    1 if team.id in self.agent.played_against else 0
                ])
        return np.array(state)
    
    # Simulations
    def simulate_tournament(self, policy:str='win_all', verbose:bool = False) -> List[Tuple[int, int, float]]:
        """
        Simulate entire tournament with naive policies :
            - all teams trying to win if policy= 'win_all'
            - the agent looses purposedly the first game in policy = 'lose_first'
        
        Returns:
            List of (team_id, points, strength) sorted by final standings
        """
        self.reset()
        
        for rd in range(self.n_rounds):
            if verbose :
                print(f"--- Simulating round n°{rd +1} ---")
            self.step(2 if (rd == 0 and policy == 'lose_first') else 0, verbose = verbose)
            
        rankings = self._get_rankings()
        team_rewards = [sum([bonus if rank <threshold else 0 for bonus, threshold in zip(self.bonus_points, self.threshold_ranks)]) + team.points for rank,team in enumerate(rankings)]
        return [(team.id, team.points, team_rewards[i],team.opponents_avg_points, team.strength) for i,team in enumerate(rankings)]

    def simulate_n_tournaments(self, n: int, policy:str='win_all', n_cores: int = None, display_results:bool=False) -> np.ndarray:
        """
        Simulate n tournaments in parallel and collect statistics.
        
        Parameters:
        -----------
        n : int
            Number of tournaments to simulate
        n_cores : int, optional
            Number of CPU cores to use. If None, uses all available cores.
        
        Returns:
        --------
        np.ndarray of shape (n_teams, 3 + len(thresholds))
            Each row corresponds to a team and contains:
            - Team strength
            - Average points
            - Average ranking
            - Share of tournaments where team ranked better than each threshold
        """
        n_teams = len(self.teams)
        thresholds = self.threshold_ranks
        n_thresholds = len(thresholds)
        
        # Create n copies of the environment for parallel processing
        env_copies = [(copy.deepcopy(self), policy) for _ in range(n)]
        
        # Run simulations in parallel with progress bar
        results = list(_imap_unordered_bar(
            _parallel_simulation, 
            env_copies, 
            n_processes=n_cores, 
            total=n,
            desc="Simulating tournaments"
        ))
        
        # Filter out failed simulations (None results)
        valid_results = [r for r in results if r is not None]
        n_valid = len(valid_results)
        
        if n_valid == 0:
            raise RuntimeError("All simulations failed. Please check your simulation parameters.")
        
        if n_valid < n:
            print(f"Warning: {n - n_valid} simulations failed out of {n} ({(n - n_valid)/n:.1%})")
        
        # Unpack results
        rankings_history = np.zeros((n_teams, n_valid))
        points_history = np.zeros((n_teams, n_valid))
        reward_history = np.zeros((n_teams, n_valid))
        
        for i, (rankings, points, reward) in enumerate(valid_results):
            rankings_history[:, i] = rankings
            points_history[:, i] = points
            reward_history[:, i] = reward
        # Calculate statistics
        results = np.zeros((n_teams, 5 + n_thresholds))
        
        for team_id in range(n_teams):
            # Column 0: Team strength
            results[team_id, 0] = self.teams[team_id].strength
            
            # Column 1: Average points
            results[team_id, 1] = np.mean(points_history[team_id])
            
            # Column 2: Average ranking
            results[team_id, 2] = np.mean(rankings_history[team_id])
            
            # Column 3: Average reward
            results[team_id, 3] = np.mean(reward_history[team_id])
            
            # Column 4: Average reward
            results[team_id, 4] = np.std(reward_history[team_id])
            
            # Columns 5+: Share of tournaments where team ranked better than each threshold
            for t_idx, threshold in enumerate(thresholds):
                results[team_id, 5 + t_idx] = np.mean(rankings_history[team_id] <= threshold)
        
        if display_results:
            print(f"\nSimulation Results (from {n_valid} tournaments):")
            print("Team | Strength | Avg Points | Avg Rank | Avg Reward | Std Reward |", end=" ")
            for t in thresholds:
                print(f"Top-{t} % |", end=" ")
            print()
            print("-" * (64 + 12 * n_thresholds))
            
            for team_id in range(n_teams):
                print(f"{team_id:4d} | {results[team_id, 0]:8.2f} | "
                      f"{results[team_id, 1]:10.2f} | {results[team_id, 2]:8.2f} | "
                      f"{results[team_id, 3]:10.2f} | {results[team_id, 4]:10.2f} |", end=" ")
                for t_idx in range(n_thresholds):
                    print(f"{results[team_id, 5 + t_idx]:7.2%} |", end=" ")
                print()
        
        result_df = pd.DataFrame(results, columns = ["Strength", "Avg_Points", "Avg_Rank", "Avg_Reward", "Std_Reward"] + [f"Top-{t} %" for t in thresholds])
        result_df = result_df.reset_index().rename(columns = {'index':'Team'})
        
        return result_df