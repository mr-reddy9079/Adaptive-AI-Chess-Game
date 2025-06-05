import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import io
import random
import chess.svg
import time
import pandas as pd
import matplotlib.pyplot as plt
import logging
from streamlit.components.v1 import html
from streamlit.components.v1 import html



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chess_app")

# Model checkpoints
ADAPTIVE_MODEL_PATH = r"C:\Users\karthi\Downloads\hybrid_checkpoint.pth"
HYPERBOLIC_MODEL_CHECKPOINT = "HuggingFaceTB/SmolLM-360M"

# =============================================
# Model Definitions
# =============================================

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, nhead=4, num_layers=2, max_len=80):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, 256)

    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
        x = self.transformer(x)
        return self.fc_out(x[0])  # Use first token representation

class ChessLLMHybridModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, cnn_channels=128, lstm_hidden=256, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cnn1 = nn.Conv1d(embed_dim, cnn_channels, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(cnn_channels * 2)
        self.lstm1 = nn.LSTM(cnn_channels * 2, lstm_hidden, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_hidden * 2, lstm_hidden, batch_first=True, bidirectional=True)
        self.fc_cnn_lstm = nn.Linear(lstm_hidden * 2, 256)
        self.llm = TinyTransformer(vocab_size)
        self.fc_final = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, llm_tokens):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.relu(self.cnn1(x))
        x = self.relu(self.batch_norm1(self.cnn2(x))).permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, 0, :]
        x = self.fc_cnn_lstm(self.dropout(x))
        llm_features = self.llm(llm_tokens.to(x.device))
        combined = torch.cat((x, llm_features), dim=1)
        return self.fc_final(self.dropout(combined))

# =============================================
# AI Classes
# =============================================

class HyperbolicChessAI:
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint
        logger.info("[HyperbolicAI] Initialized Hyperbolic AI")
        self.temperature = 1.2
        self.risk_factor = 0.7
        self.style = "hyper-aggressive"
        
    def evaluate_board(self, board):
        piece_values = {
            chess.PAWN: 1.0, chess.KNIGHT: 3.5, chess.BISHOP: 3.3,
            chess.ROOK: 5.5, chess.QUEEN: 10.0, chess.KING: 0.0
        }
        evaluation = 0.0
        attack_bonus = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                mobility = len(board.attacks(square))
                value += mobility * 0.05 * self.temperature
                attacked_value = 0
                for attacked_square in board.attacks(square):
                    attacked_piece = board.piece_at(attacked_square)
                    if attacked_piece and attacked_piece.color != piece.color:
                        attacked_value += piece_values[attacked_piece.piece_type] * 0.1
                attack_bonus += attacked_value * self.risk_factor
                if piece.color == chess.WHITE:
                    evaluation += value
                else:
                    evaluation -= value
        evaluation = np.sign(evaluation) * (abs(evaluation) ** 1.2)
        evaluation += attack_bonus
        return evaluation
    
    def get_move(self, board, time_limit=5.0):
        start_time = time.time()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        complexity = self.calculate_position_complexity(board)
        time_factor = min(1.0, complexity / 20.0)
        remaining_time = time_limit * (1.0 + time_factor)
        best_move = None
        best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
        for move in legal_moves:
            if time.time() - start_time > remaining_time:
                break
            board.push(move)
            score = self.evaluate_board(board)
            score = self.apply_hyperbolic_transform(score, board.turn)
            score += random.uniform(-0.5, 0.5) * self.temperature
            board.pop()
            if (board.turn == chess.WHITE and score > best_score) or (board.turn == chess.BLACK and score < best_score):
                best_score = score
                best_move = move
        return best_move if best_move else random.choice(legal_moves)
    
    def calculate_position_complexity(self, board):
        complexity = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                complexity += 1
                complexity += len(board.attacks(square)) * 0.2
        return complexity
    
    def apply_hyperbolic_transform(self, score, color):
        if color == chess.WHITE:
            return np.sign(score) * (abs(score) ** 1.3)
        else:
            return np.sign(score) * (abs(score) ** 1.3)

class AdaptiveChessAI:
    def __init__(self, model, device, vocab, pad_token):
        self.model = model
        self.device = device
        self.vocab = vocab
        self.pad_token = pad_token
        self.opponent_skill_estimate = 5
        self.opponent_style = "unknown"
        self.adaptive_factor = 0.8
        self.history = []
        self.piece_values = {
            chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
            chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0
        }
        
    def update_opponent_assessment(self, board, opponent_move, time_taken):
        if not opponent_move:
            return
        quality_score = self._analyze_move_quality(board, opponent_move)
        style_score = self._analyze_move_style(board, opponent_move)
        skill_adjustment = quality_score * (1.0 / (time_taken + 0.5))
        self.opponent_skill_estimate = (1 - self.adaptive_factor) * self.opponent_skill_estimate + \
                                      self.adaptive_factor * min(10, max(1, skill_adjustment * 10))
        if style_score > 0.6:
            self.opponent_style = "aggressive"
        elif style_score < -0.6:
            self.opponent_style = "defensive"
        else:
            self.opponent_style = "balanced"
        logger.info(f"Updated opponent assessment: Skill={self.opponent_skill_estimate:.1f}, Style={self.opponent_style}")
        self.history.append({
            'skill_estimate': self.opponent_skill_estimate,
            'style': self.opponent_style,
            'quality_score': quality_score,
            'style_score': style_score
        })
    
    def _analyze_move_quality(self, board, move):
        if move not in board.legal_moves:
            return 0.1
        board_copy = board.copy()
        try:
            board_copy.push(move)
        except AssertionError:
            return 0.1
        quality = 0.5
        if board_copy.is_check():
            quality += 0.3
        if board_copy.is_checkmate():
            quality = 1.0
            return quality
        elif board.is_capture(move):
            quality += 0.2
        center_control = self._assess_center_control(board_copy)
        piece_development = self._assess_piece_development(board_copy)
        quality += (center_control + piece_development) / 4
        undefended_penalty = self._calculate_undefended_penalty(board_copy)
        quality -= undefended_penalty
        return min(1.0, max(0.1, quality))
    
    def _calculate_undefended_penalty(self, board):
        penalty = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attackers = board.attackers(not board.turn, square)
                defenders = board.attackers(board.turn, square)
                if attackers and len(defenders) < len(attackers):
                    penalty += 0.05 * self.piece_values.get(piece.piece_type, 1)
        return min(0.3, penalty)
    
    def _analyze_move_style(self, board, move):
        if move not in board.legal_moves:
            return 0.0
        board_copy = board.copy()
        try:
            board_copy.push(move)
        except AssertionError:
            return 0.0
        attack_score = 0
        defense_score = 0
        for square in chess.SQUARES:
            piece = board_copy.piece_at(square)
            if piece and piece.color == board_copy.turn:
                for target in board_copy.attacks(square):
                    target_piece = board_copy.piece_at(target)
                    if target_piece and target_piece.color != board_copy.turn:
                        attack_score += 1
                for own_square in chess.SQUARES:
                    own_piece = board_copy.piece_at(own_square)
                    if own_piece and own_piece.color == board_copy.turn:
                        if square in board_copy.attackers(board_copy.turn, own_square):
                            defense_score += 1
        if attack_score + defense_score == 0:
            return 0.0
        return (attack_score - defense_score) / max(1, attack_score + defense_score)
    
    def _assess_center_control(self, board):
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        control = 0
        for square in center_squares:
            if board.piece_at(square) and board.piece_at(square).color == board.turn:
                control += 0.25
            elif board.attackers(board.turn, square):
                control += 0.15
        return min(1.0, control)
    
    def _assess_piece_development(self, board):
        developed = 0
        total_pieces = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                total_pieces += 1
                if piece.piece_type == chess.PAWN:
                    continue
                rank = chess.square_rank(square)
                if (piece.color == chess.WHITE and rank > 0) or \
                   (piece.color == chess.BLACK and rank < 7):
                    developed += 1
        return developed / max(1, total_pieces - 8)
    
    def evaluate_board(self, board):
        if self.model is None:
            eval_score = 0.0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = self.piece_values[piece.piece_type]
                    if piece.color == chess.WHITE:
                        eval_score += value
                    else:
                        eval_score -= value
            return eval_score
        fen = board.fen()
        tokens = self._fen_to_tokens(fen)
        token_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        llm_tokens = token_tensor.clone()
        with torch.no_grad():
            outputs = self.model(token_tensor, llm_tokens)
            eval_score = 0.0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = self.piece_values[piece.piece_type]
                    if piece.color == chess.WHITE:
                        eval_score += value
                    else:
                        eval_score -= value
            if self.opponent_style == "aggressive":
                defense_bonus = self._calculate_defense_bonus(board)
                eval_score += defense_bonus if board.turn == chess.BLACK else -defense_bonus
            elif self.opponent_style == "defensive":
                attack_bonus = self._calculate_attack_bonus(board)
                eval_score += attack_bonus if board.turn == chess.BLACK else -attack_bonus
            return eval_score
    
    def _calculate_defense_bonus(self, board):
        defense_score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                defenders = len(board.attackers(board.turn, square))
                defense_score += defenders * 0.05
        return defense_score
    
    def _calculate_attack_bonus(self, board):
        attack_score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                attackers = len(board.attackers(board.turn, square))
                attack_score += attackers * 0.08
        return attack_score
        
    def _fen_to_tokens(self, fen):
        max_len = 80
        tokens = [self.vocab.get(ch, self.pad_token) for ch in fen]
        return tokens[:max_len] + [self.pad_token] * (max_len - len(tokens))
        
    def get_skill_level(self):
        adjustment = random.uniform(-0.5, 1.0)
        skill = min(10, max(1, self.opponent_skill_estimate + adjustment))
        if len(self.history) > 10:
            skill = min(10, skill + 1)
        return skill

    def get_move(self, board):
        skill_level = self.get_skill_level()
        logger.info(f"Adaptive AI playing at skill level: {skill_level:.1f}")
        depth = max(1, min(3, int(skill_level) // 3))
        best_move = None
        best_value = -float('inf') if board.turn == chess.WHITE else float('inf')
        alpha, beta = -float('inf'), float('inf')
        legal_moves = list(board.legal_moves)
        if skill_level < 5:
            legal_moves = random.sample(legal_moves, min(5, len(legal_moves)))
        for move in legal_moves:
            board.push(move)
            eval = self._minimax(board, depth-1, alpha, beta, board.turn == chess.BLACK, skill_level)
            board.pop()
            if board.turn == chess.WHITE:
                if eval > best_value:
                    best_value = eval
                    best_move = move
                alpha = max(alpha, eval)
            else:
                if eval < best_value:
                    best_value = eval
                    best_move = move
                beta = min(beta, eval)
        return best_move if best_move else random.choice(list(board.legal_moves))
    
    def _minimax(self, board, depth, alpha, beta, maximizing_player, skill_level):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        legal_moves = list(board.legal_moves)
        if skill_level < 5 and random.random() < 0.3:
            random.shuffle(legal_moves)
        if maximizing_player:
            max_eval = -float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self._minimax(board, depth-1, alpha, beta, False, skill_level)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self._minimax(board, depth-1, alpha, beta, True, skill_level)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

# =============================================
# Utility Functions
# =============================================

@st.cache_resource
def load_adaptive_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading adaptive model on {device}")
    vocab = {
        '<PAD>': 0, 'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6,
        'p': 7, 'r': 8, 'n': 9, 'b': 10, 'q': 11, 'k': 12, '1': 13,
        '2': 14, '3': 15, '4': 16, '5': 17, '6': 18, '7': 19, '8': 20,
        '/': 21, ' ': 22, 'w': 23, 'b': 24, 'K': 25, 'Q': 26, 'k': 27,
        'q': 28, '-': 29, '0': 30, 'a': 31, 'c': 32, 'd': 33
    }
    pad_token = len(vocab) + 1
    vocab_size = 34
    model = ChessLLMHybridModel(vocab_size=vocab_size, num_classes=10).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info("Adaptive model loaded successfully")
        return model, device, vocab, pad_token
    except Exception as e:
        logger.error(f"Error loading adaptive model: {e}")
        st.error(f"Failed to load adaptive model: {e}")
        return None, device, vocab, pad_token


def svg_to_image(svg_string):
    # Directly render the SVG in the browser
    html(svg_string, height=400)
    return None

def fen_to_tokens(fen, vocab, pad_token, max_len=80):
    tokens = [vocab.get(ch, pad_token) for ch in fen]
    return tokens[:max_len] + [pad_token] * (max_len - len(tokens))

def evaluate_board(board, model, device, vocab, pad_token):
    if model is None:
        piece_values = {
            chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
            chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0
        }
        eval_score = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    eval_score += value
                else:
                    eval_score -= value
        return eval_score, 5
    fen = board.fen()
    tokens = fen_to_tokens(fen, vocab, pad_token)
    token_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    llm_tokens = token_tensor.clone()
    with torch.no_grad():
        outputs = model(token_tensor, llm_tokens)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        skill_idx = torch.argmax(probabilities, dim=1).item()
        skill_level = skill_idx + 1
        piece_values = {
            chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
            chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0
        }
        eval_score = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    eval_score += value
                else:
                    eval_score -= value
        return eval_score, skill_level

def minimax(board, depth, alpha, beta, maximizing_player, skill_level, model, device, vocab, pad_token):
    if depth == 0 or board.is_game_over():
        eval_score, _ = evaluate_board(board, model, device, vocab, pad_token)
        return eval_score
    legal_moves = list(board.legal_moves)
    if skill_level < 5 and random.random() < 0.3:
        random.shuffle(legal_moves)
    if maximizing_player:
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth-1, alpha, beta, False, skill_level, model, device, vocab, pad_token)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth-1, alpha, beta, True, skill_level, model, device, vocab, pad_token)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_ai_move(board, skill_level, model, device, vocab, pad_token):
    if model is None:
        return random.choice(list(board.legal_moves))
    if skill_level <= 3:
        depth = 3
    elif skill_level == 4:
        depth = 4
    elif skill_level == 5:
        depth = 5
    elif skill_level == 6:
        depth = 6
    elif skill_level <= 8:
        depth = 8
    else:
        depth = 10
    if len(board.piece_map()) > 20:
        depth = max(2, depth - 2)
    elif len(board.piece_map()) > 10:
        depth = max(1, depth - 1)
    best_move = None
    best_value = -float('inf') if board.turn == chess.WHITE else float('inf')
    alpha, beta = -float('inf'), float('inf')
    legal_moves = list(board.legal_moves)
    if skill_level < 5:
        legal_moves = random.sample(legal_moves, min(5, len(legal_moves)))
    elif skill_level < 8:
        legal_moves = random.sample(legal_moves, min(15, len(legal_moves)))
    if skill_level >= 5:
        legal_moves.sort(key=lambda m: move_heuristic(board, m), reverse=board.turn == chess.WHITE)
    for move in legal_moves:
        board.push(move)
        eval = minimax(board, depth-1, alpha, beta, board.turn == chess.BLACK, skill_level, model, device, vocab, pad_token)
        board.pop()
        if board.turn == chess.WHITE:
            if eval > best_value:
                best_value = eval
                best_move = move
            alpha = max(alpha, eval)
        else:
            if eval < best_value:
                best_value = eval
                best_move = move
            beta = min(beta, eval)
        if (board.turn == chess.WHITE and best_value >= 10000) or (board.turn == chess.BLACK and best_value <= -10000):
            break
    return best_move if best_move else random.choice(list(board.legal_moves))

def move_heuristic(board, move):
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            piece_value = {'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1, 'q': 9, 'r': 5, 'b': 3, 'n': 3, 'p': 1}.get(captured_piece.symbol(), 0)
            return 10 + piece_value
        return 10
    if move.promotion:
        return 15
    if board.gives_check(move):
        return 5
    return 0

def get_available_pieces(board):
    pieces = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == board.turn:
            pieces.append((chess.square_name(square), piece.symbol()))
    return pieces

def get_possible_moves(board, square_name):
    square = chess.parse_square(square_name)
    return [move for move in board.legal_moves if move.from_square == square]

def plot_game_analysis(move_history, mode):
    fig = plt.figure(figsize=(12, 8))
    if mode == "Human vs Adaptive AI":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        move_numbers = [i+1 for i in range(len(move_history))]
        eval_scores = [move['eval_score'] for move in move_history]
        ax1.plot(move_numbers, eval_scores, 'b-', marker='o')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_title('Material Balance Over Time')
        ax1.set_xlabel('Move Number')
        ax1.set_ylabel('Evaluation Score')
        ax1.grid(True, alpha=0.3)
        skill_levels = [move['skill_level'] for move in move_history if 'skill_level' in move]
        if skill_levels:
            ax2.plot(range(1, len(skill_levels)+1), skill_levels, 'r-', marker='o')
            ax2.set_title('Estimated Player Skill Level')
            ax2.set_xlabel('Move Number')
            ax2.set_ylabel('Skill Level (1-10)')
            ax2.set_ylim(0, 11)
            ax2.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        move_numbers = [i+1 for i in range(len(move_history))]
        white_scores = [move['white_eval'] for move in move_history]
        black_scores = [-move['black_eval'] for move in move_history]
        ax1.plot(move_numbers, white_scores, 'b-', marker='o', label='White (Hyperbolic AI)')
        ax1.plot(move_numbers, black_scores, 'r-', marker='o', label='Black (Adaptive AI)')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_title('Material Balance Over Time')
        ax1.set_xlabel('Move Number')
        ax1.set_ylabel('Evaluation Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        if 'time_taken' in move_history[0]:
            white_times = [move['time_taken'] if i % 2 == 0 else 0 for i, move in enumerate(move_history)]
            black_times = [move['time_taken'] if i % 2 == 1 else 0 for i, move in enumerate(move_history)]
            ax2.bar(move_numbers, white_times, color='b', alpha=0.6, label='White Move Time')
            ax2.bar(move_numbers, black_times, color='r', alpha=0.6, label='Black Move Time')
            ax2.set_title('Move Time Analysis')
            ax2.set_xlabel('Move Number')
            ax2.set_ylabel('Time (seconds)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        if 'adaptive_skill' in move_history[0]:
            skill_levels = [move.get('adaptive_skill', 5) for move in move_history]
            ax3.plot(move_numbers, skill_levels, 'g-', marker='o')
            ax3.set_title('Adaptive AI Skill Level Adjustment')
            ax3.set_xlabel('Move Number')
            ax3.set_ylabel('Skill Level (1-10)')
            ax3.set_ylim(0, 11)
            ax3.grid(True, alpha=0.3)
        plt.tight_layout()
    return fig

def analyze_game_result(board, move_history):
    analysis = {
        "total_moves": len(move_history),
        "white_captures": sum(1 for move in move_history if move['player'] == 'Hyperbolic AI' and board.is_capture(move['move'])),
        "black_captures": sum(1 for move in move_history if move['player'] == 'Adaptive AI' and board.is_capture(move['move'])),
        "white_avg_time": np.mean([move.get('time_taken', 0) for move in move_history if move['player'] == 'Hyperbolic AI']),
        "black_avg_time": np.mean([move.get('time_taken', 0) for move in move_history if move['player'] == 'Adaptive AI']),
        "final_result": board.result()
    }
    eval_swings = []
    for i in range(1, len(move_history)):
        swing = abs(move_history[i]['white_eval'] - move_history[i-1]['white_eval'])
        eval_swings.append(swing)
    analysis["max_swing"] = max(eval_swings) if eval_swings else 0
    analysis["avg_swing"] = np.mean(eval_swings) if eval_swings else 0
    return analysis

# =============================================
# Main Streamlit App
# =============================================

def main():
    st.set_page_config(page_title="Chess AI Battle", layout="wide")
    st.title("♟️ Chess AI Battle")

    # Initialize session state
    if 'board' not in st.session_state:
        st.session_state.board = chess.Board()
    if 'move_history' not in st.session_state:
        st.session_state.move_history = []
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
    if 'selected_piece' not in st.session_state:
        st.session_state.selected_piece = None
    if 'possible_moves' not in st.session_state:
        st.session_state.possible_moves = []
    if 'player_skill' not in st.session_state:
        st.session_state.player_skill = 5
    if 'adaptive_model' not in st.session_state:
        st.session_state.adaptive_model = None
    if 'hyperbolic_ai' not in st.session_state:
        st.session_state.hyperbolic_ai = None
    if 'adaptive_ai' not in st.session_state:
        st.session_state.adaptive_ai = None

    # Load models
    if st.session_state.adaptive_model is None:
        with st.spinner("Loading Adaptive AI..."):
            model, device, vocab, pad_token = load_adaptive_model(ADAPTIVE_MODEL_PATH)
            st.session_state.adaptive_model = (model, device, vocab, pad_token)
            if model is not None:
                st.session_state.adaptive_ai = AdaptiveChessAI(model, device, vocab, pad_token)
    
    if st.session_state.hyperbolic_ai is None:
        with st.spinner("Initializing Hyperbolic AI..."):
            st.session_state.hyperbolic_ai = HyperbolicChessAI(HYPERBOLIC_MODEL_CHECKPOINT)

    # Sidebar controls
    with st.sidebar:
        st.header("Game Mode")
        game_mode = st.selectbox(
            "Select Game Mode:",
            ["Human vs Adaptive AI", "Adaptive AI vs Hyperbolic AI"],
            key="game_mode"
        )
        
        st.header("Game Controls")
        if st.button("Start Game") and not st.session_state.game_active:
            st.session_state.board = chess.Board()
            st.session_state.move_history = []
            st.session_state.game_active = True
            st.session_state.selected_piece = None
            st.session_state.possible_moves = []
            st.session_state.player_skill = 5
            if st.session_state.adaptive_ai:
                st.session_state.adaptive_ai.opponent_skill_estimate = 5
                st.session_state.adaptive_ai.opponent_style = "unknown"
                st.session_state.adaptive_ai.history = []
            st.rerun()
        
        if st.button("Stop Game") and st.session_state.game_active:
            st.session_state.game_active = False
            st.rerun()

        if game_mode == "Adaptive AI vs Hyperbolic AI":
            st.markdown("### AI Settings")
            st.slider("Hyperbolic AI Temperature", 0.1, 2.0, 1.2, key="hyperbolic_temp")
            if st.session_state.adaptive_ai:
                current_skill = st.session_state.adaptive_ai.get_skill_level()
                st.info(f"Current Adaptive AI Level: {current_skill:.1f}/10")
                st.caption("Level automatically adjusts based on opponent")

    # Main game area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chess board
        svg = chess.svg.board(
            board=st.session_state.board,
            size=400,
            squares=chess.SquareSet([chess.parse_square(st.session_state.selected_piece)]) if st.session_state.selected_piece else None,
            lastmove=st.session_state.board.peek() if st.session_state.board.move_stack else None
        )
        svg_to_image(svg)  # This now renders directly with HTML

        # Game logic based on mode
        if game_mode == "Human vs Adaptive AI":
            model, device, vocab, pad_token = st.session_state.adaptive_model or (None, torch.device("cpu"), {}, 0)
            
            if st.session_state.board.turn == chess.WHITE:
                st.write("White's turn (You)")
            else:
                st.write("Black's turn (AI)")
                
            if not st.session_state.board.is_game_over() and st.session_state.game_active:
                if st.session_state.board.turn == chess.WHITE:
                    available_pieces = get_available_pieces(st.session_state.board)
                    piece_options = [f"{name} ({symbol})" for name, symbol in available_pieces]
                    selected_piece = st.selectbox(
                        "Select your piece:",
                        options=piece_options,
                        index=0,
                        key="piece_select"
                    )
                    if selected_piece:
                        square_name = selected_piece.split()[0]
                        st.session_state.selected_piece = square_name
                        st.session_state.possible_moves = get_possible_moves(st.session_state.board, square_name)
                        if st.session_state.possible_moves:
                            move_options = [move.uci() for move in st.session_state.possible_moves]
                            selected_move = st.selectbox(
                                "Select move:",
                                options=move_options,
                                index=0,
                                key="move_select"
                            )
                            if st.button("Make Move"):
                                try:
                                    move = chess.Move.from_uci(selected_move)
                                    if move in st.session_state.board.legal_moves:
                                        move_san = st.session_state.board.san(move)
                                        st.session_state.board.push(move)
                                        eval_score, skill_level = evaluate_board(st.session_state.board, model, device, vocab, pad_token)
                                        st.session_state.player_skill = skill_level
                                        st.session_state.move_history.append({
                                            'move': selected_move,
                                            'san': move_san,
                                            'eval_score': eval_score,
                                            'skill_level': skill_level
                                        })
                                        st.session_state.selected_piece = None
                                        st.session_state.possible_moves = []
                                        st.rerun()
                                    else:
                                        st.error("Illegal move! Please try again.")
                                except Exception as e:
                                    st.error(f"Error making move: {e}")
                        else:
                            st.warning("No legal moves for selected piece")
                else:
                    if st.button("Make AI Move"):
                        with st.spinner(f"AI thinking (Skill level: {st.session_state.player_skill})..."):
                            ai_move = get_ai_move(
                                st.session_state.board,
                                st.session_state.player_skill,
                                model,
                                device,
                                vocab,
                                pad_token
                            )
                            if ai_move and ai_move in st.session_state.board.legal_moves:
                                move_san = st.session_state.board.san(ai_move)
                                st.session_state.board.push(ai_move)
                                eval_score, _ = evaluate_board(st.session_state.board, model, device, vocab, pad_token)
                                st.session_state.move_history.append({
                                    'move': ai_move.uci(),
                                    'san': move_san,
                                    'eval_score': eval_score
                                })
                                st.rerun()
                            else:
                                st.error("AI couldn't find a legal move!")
            elif st.session_state.board.is_game_over():
                st.session_state.game_active = False
                result = st.session_state.board.result()
                if result == "1-0":
                    st.success("White wins!")
                elif result == "0-1":
                    st.success("Black wins!")
                else:
                    st.success("Draw!")
                st.subheader("Game Analysis")
                fig = plot_game_analysis(st.session_state.move_history, game_mode)
                st.pyplot(fig)
                if st.button("Play Again"):
                    st.session_state.board = chess.Board()
                    st.session_state.move_history = []
                    st.session_state.selected_piece = None
                    st.session_state.possible_moves = []
                    st.session_state.player_skill = 5
                    st.session_state.game_active = True
                    st.rerun()
            else:
                st.info("Click 'Start Game' to begin")

        else:  # Adaptive AI vs Hyperbolic AI
            if st.session_state.game_active:
                if st.session_state.board.is_game_over():
                    st.session_state.game_active = False
                    result = st.session_state.board.result()
                    if result == "1-0":
                        st.success("Hyperbolic AI (White) wins!")
                    elif result == "0-1":
                        st.success("Adaptive AI (Black) wins!")
                    else:
                        st.success("Draw!")
                    analysis = analyze_game_result(st.session_state.board, st.session_state.move_history)
                    st.subheader("Battle Analysis")
                    st.write(f"Total moves: {analysis['total_moves']}")
                    st.write(f"Hyperbolic AI captures: {analysis['white_captures']}")
                    st.write(f"Adaptive AI captures: {analysis['black_captures']}")
                    st.write(f"Largest evaluation swing: {analysis['max_swing']:.2f}")
                    st.write(f"Average evaluation swing: {analysis['avg_swing']:.2f}")
                    fig = plot_game_analysis(st.session_state.move_history, game_mode)
                    st.pyplot(fig)
                else:
                    current_player = "Hyperbolic AI (White)" if st.session_state.board.turn == chess.WHITE else "Adaptive AI (Black)"
                    st.write(f"{current_player} is thinking...")
                    if st.session_state.board.turn == chess.WHITE:
                        hyperbolic_ai = st.session_state.hyperbolic_ai
                        hyperbolic_ai.temperature = st.session_state.hyperbolic_temp
                        start_time = time.time()
                        move = hyperbolic_ai.get_move(st.session_state.board)
                        time_taken = time.time() - start_time
                        if move and move in st.session_state.board.legal_moves:
                            move_san = st.session_state.board.san(move)
                            st.session_state.board.push(move)
                            if st.session_state.adaptive_ai:
                                st.session_state.adaptive_ai.update_opponent_assessment(
                                    st.session_state.board, move, time_taken
                                )
                            white_eval = hyperbolic_ai.evaluate_board(st.session_state.board)
                            black_eval = st.session_state.adaptive_ai.evaluate_board(st.session_state.board) \
                                        if st.session_state.adaptive_ai else 0
                            st.session_state.move_history.append({
                                'move': move,
                                'san': move_san,
                                'white_eval': white_eval,
                                'black_eval': black_eval,
                                'time_taken': time_taken,
                                'player': 'Hyperbolic AI'
                            })
                            st.rerun()
                    else:
                        adaptive_ai = st.session_state.adaptive_ai
                        start_time = time.time()
                        move = adaptive_ai.get_move(st.session_state.board)
                        time_taken = time.time() - start_time
                        if move and move in st.session_state.board.legal_moves:
                            move_san = st.session_state.board.san(move)
                            st.session_state.board.push(move)
                            current_skill = adaptive_ai.get_skill_level()
                            white_eval = st.session_state.hyperbolic_ai.evaluate_board(st.session_state.board)
                            black_eval = adaptive_ai.evaluate_board(st.session_state.board)
                            st.session_state.move_history.append({
                                'move': move,
                                'san': move_san,
                                'white_eval': white_eval,
                                'black_eval': black_eval,
                                'time_taken': time_taken,
                                'player': 'Adaptive AI',
                                'adaptive_skill': current_skill
                            })
                            st.rerun()
            else:
                st.info("Click 'Start Game' to begin the AI vs AI match")

    with col2:
        st.subheader("Game Status")
        if st.session_state.move_history:
            last_move = st.session_state.move_history[-1]
            st.write(f"Last move: {last_move['san']} by {last_move['player'] if game_mode == 'Adaptive AI vs Hyperbolic AI' else ('You' if st.session_state.board.turn == chess.BLACK else 'AI')}")
            if game_mode == "Adaptive AI vs Hyperbolic AI":
                st.write(f"Time taken: {last_move.get('time_taken', 0):.2f}s")
                st.write("### Current Evaluations")
                st.write(f"Hyperbolic AI (White) evaluation: {last_move['white_eval']:.2f}")
                st.write(f"Adaptive AI (Black) evaluation: {last_move['black_eval']:.2f}")
                if st.session_state.adaptive_ai and 'adaptive_skill' in last_move:
                    st.write("### Adaptive AI Status")
                    st.write(f"Current skill level: {last_move['adaptive_skill']:.1f}")
                    st.write(f"Playing style: {st.session_state.adaptive_ai.opponent_style}")
            else:
                model, device, vocab, pad_token = st.session_state.adaptive_model or (None, torch.device("cpu"), {}, 0)
                eval_score, skill_level = evaluate_board(st.session_state.board, model, device, vocab, pad_token)
                st.write(f"Material balance: {eval_score:.2f} ({'White' if eval_score > 0 else 'Black'} advantage)")
                st.write(f"Estimated player skill: {skill_level}/10")
            
            st.write("### Move History")
            moves_text = ""
            for i, move_info in enumerate(st.session_state.move_history):
                if i % 2 == 0:
                    moves_text += f"{i//2 + 1}. {move_info['san']} "
                else:
                    moves_text += f"{move_info['san']}\n"
            st.text(moves_text)
        else:
            st.write("No moves yet")

if __name__ == "__main__":
    main()
