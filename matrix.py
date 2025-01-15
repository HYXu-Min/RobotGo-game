import enum
import random
from collections import namedtuple
import copy
import math
import numpy as np
import pygame


class Player(enum.Enum):
    black = 1
    white = 2

    '''
    返回对方棋子颜色，如果本方是白棋，那就返回Player.black
    '''

    @property
    def other(self):
        if self == Player.white:
            return Player.black
        else:
            return Player.white


class Point(namedtuple('Point', 'row col')):
    def neighbors(self):
        '''
        返回当前点的相邻点，也就是相对于当前点的上下左右四个点
        '''
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]


class Move():
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        # 是否轮到我下
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, point):
        return Move(point=point)

    @classmethod
    # 让对方继续下
    def pass_turn(cls):
        return Move(is_pass=True)

    @classmethod
    # 投子认输
    def resign(cls):
        return Move(is_resign=True)


class GoString():
    def __init__(self, color, stones, liberties):
        self.color = color  # 黑/白
        # 将两个集合修改为immutable类型
        self.stones = frozenset(stones)  # stone就是棋子
        self.liberties = frozenset(liberties)  # 自由点

    # 替换掉原来的remove_liberty 和 add_liberty
    def without_liberty(self, point):
        new_liberties = self.liberties - set([point])
        return GoString(self.color, self.stones, new_liberties)

    def with_liberty(self, point):
        new_liberties = self.liberties | set([point])
        return GoString(self.color, self.stones, new_liberties)

    def merged_with(self, go_string):
        # 落子之后，两片相邻棋子可能会合成一片
        '''
        假设*代表黑棋，o代表白棋，x代表没有落子的棋盘点，当前棋盘如下：
        x  x  x  x  x  x
        x  *  x! *  o  *
        x  x  x  *  o  x
        x  x  *  x  o  x
        x  x  *  o  x  x
        注意看带!的x，如果我们把黑子下在那个地方，那么x!左边的黑棋和新下的黑棋会调用当前函数进行合并，
        同时x!上方的x和下面的x就会成为合并后相邻棋子共同具有的自由点。同时x!原来属于左边黑棋的自由点，
        现在被一个黑棋占据了，所以下面代码要把该点从原来的自由点集合中去掉
        '''
        assert go_string.color == self.color
        combined_stones = self.stones | go_string.stones
        return GoString(self.color, combined_stones,
                        (self.liberties | go_string.liberties) - combined_stones)

    @property
    def num_liberties(self):  # 自由点的数量
        return len(self.liberties)

    def __eq__(self, other):  # 是否相等
        return isinstance(other,
                          GoString) and self.color == other.color and self.stones == other.stones and self.liberties == other.liberties


# 实现棋盘
class Board():
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}
        # 添加hash
        self._hash = zobrist_EMPTY_BOARD

    def zobrist_hash(self):
        return self._hash

    def place_stone(self, player, point):
        # 确保位置在棋盘内
        assert self.is_on_grid(point)
        # 确定给定位置没有被占据
        assert self._grid.get(point) is None

        adjacent_same_color = []
        adjacent_opposite_color = []
        liberties = []

        for neighbor in point.neighbors():
            # 判断落子点上下左右的邻接点情况
            if not self.is_on_grid(neighbor):
                continue

            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                # 如果邻接点没有被占据，那么就是当前落子点的自由点
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    # 记录与棋子同色的连接棋子
                    adjacent_same_color.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    # 记录落点邻接点与棋子不同色的棋子
                    adjacent_opposite_color.append(neighbor_string)

        # 将当前落子与棋盘上相邻的棋子合并成一片
        new_string = GoString(player, [point], liberties)

        # 从下面开始新的修改
        for same_color_string in adjacent_same_color:
            new_string = new_string.merged_with(same_color_string)
        for new_string_point in new_string.stones:
            # 访问棋盘某个点时返回与该点棋子相邻的所有棋子集合
            self._grid[new_string_point] = new_string

        # 增加落子的hash值记录
        self._hash ^= zobrist_HASH_CODE[point, None]
        self._hash ^= zobrist_HASH_CODE[point, player]

        for other_color_string in adjacent_opposite_color:
            # 当该点被占据前，它属于反色棋子的自由点，占据后就不再属于反色棋子自由点
            # 修改成without_liberty
            replacement = other_color_string.without_liberty(point)
            if replacement.num_liberties:
                self._replace_string(other_color_string.without_liberty(point))

            else:
                # 如果落子后，相邻反色棋子的所有自由点都被堵住，对方棋子被吃掉
                self._remove_string(other_color_string)

    # 增加一个新函数
    def _replace_string(self, new_string):
        for point in new_string.stones:
            self._grid[point] = new_string

    def is_on_grid(self, point):
        return 1 <= point.row <= self.num_rows and 1 <= point.col <= self.num_cols

    def get(self, point):
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    def get_go_string(self, point):
        string = self._grid.get(point)
        if string is None:
            return None
        return string

    def _remove_string(self, string):
        # 从棋盘上删除一整片连接棋子
        for point in string.stones:
            for neighbor in point.neighbors():
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    # 修改
                    self._replace_string(neighbor_string.with_liberty(point))

            self._grid[point] = None

            # 由于棋子被拿掉后，对应位置状态发生变化，因此修改编码
            self._hash ^= zobrist_HASH_CODE[point, string.color]
            self._hash ^= zobrist_HASH_CODE[point, None]


# 棋盘状态的检测和落子检测
class GameState():
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move

        # 添加新修改
        if previous is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(previous.previous_states | {(previous.next_player,
                                                                          previous.board.zobrist_hash())})

    def apply_move(self, move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board

        return GameState(next_board, self.next_player.other, self, move)

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)

        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    def is_over(self):
        if self.last_move is None:
            return False
        if self.last_move.is_resign:
            return True

        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False

        # 如果两个棋手同时放弃落子，棋局结束
        return self.last_move.is_pass and second_last_move.is_pass

    @property
    def situation(self):
        return (self.next_player, self.board)

    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False

        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board)

        # 判断Ko不仅仅看是否返回上一步的棋盘而是检测是否返回以前有过的棋盘状态
        # 修改,我们不用在循环检测，只要看当前数值与前面数值是否匹配即可
        return next_situation in self.previous_states

    def is_move_self_capture(self, player, move):
        if not move.is_play:
            return False

        next_board = copy.deepcopy(self.board)

        # 先落子，完成吃子后再判断是否是自己吃自己
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)

        # new_string = self.board._grid.get(move.point)
        return new_string.num_liberties == 0

    def is_valid_move(self, move):
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        # return (self.board.get(move.point) is None and
        #         not self.does_move_violate_ko(self.next_player, move) and
        #         not self.is_move_self_capture(self.next_player, move))
        return (self.board.get(move.point) is None and
                not self.does_move_violate_ko(self.next_player, move))

    def is_agent_valid(self, move):
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        return (self.board.get(move.point) is None and
                not self.does_move_violate_ko(self.next_player, move) and
                not self.is_move_self_capture(self.next_player, move))


    def legal_moves(self):
        moves = []
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play(Point(row, col))
                if self.is_agent_valid(move):
                    moves.append(move)

        return moves

    def winner(self):
        black_num = 0
        white_num = 0

        for r in range(1, self.board.num_rows):
            for c in range(1, self.board.num_cols):
                point = Point(row=r, col=c)
                color = self.board.get(point)
                if color == Player.black:
                    black_num += 1
                elif color == Player.white:
                    white_num += 1
        diff = black_num - white_num

        if diff > 0:
            return Player.black
        elif diff <= 0:
            return Player.white


def is_point_an_eye(board, point, color):
    if board.get(point) is not None:
        return False

    for neighbor in point.neighbors():
        # 检测邻接点全是己方棋子
        if board.is_on_grid(neighbor):
            neighbor_color = board.get(neighbor)
            if neighbor_color != color:
                return False
    # 四个对角线位置至少有三个被己方棋子占据
    friendly_corners = 0
    off_board_corners = 0
    corners = [
        Point(point.row - 1, point.col - 1),
        Point(point.row - 1, point.col + 1),
        Point(point.row + 1, point.col - 1),
        Point(point.row + 1, point.col + 1)
    ]
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4
    return friendly_corners >= 3


class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplementedError()


class RandomBot(Agent):
    def select_move(self, game_state):
        '''
        遍历棋盘，只要看到一个不违反规则的位置就落子
        '''
        candidates = []
        for r in range(1, game_state.board.num_rows + 1):
            for c in range(1, game_state.board.num_cols + 1):
                candidate = Point(row=r, col=c)
                if game_state.is_valid_move(Move.play(candidate)) and not \
                        is_point_an_eye(game_state.board,
                                        candidate,
                                        game_state.next_player):
                    candidates.append(candidate)
        if not candidates:
            return Move.pass_turn()

        # 在所有可选位置随便选一个
        return Move.play(random.choice(candidates))


class MCTSNode(object):
    '''
    描述蒙特卡洛树节点
    '''

    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        # parent指向根节点
        self.parent = parent
        # move表示当前节点选择的落子步骤
        self.move = move
        # 统计胜率
        self.win_counts = {
            Player.black: 0,
            Player.white: 0
        }
        # 记录胜负总数
        self.num_rollouts = 0
        self.children = []
        # 记录当前可行的所有步骤
        self.unvisited_moves = game_state.legal_moves()

    def add_random_child(self):
        # 从所有可行走法中随机选取一种
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        # 根据选择走法形成新的棋盘
        new_game_state = self.game_state.apply_move(new_move)
        # 形成子节点
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node

    def record_win(self, winner):
        # 统计胜率
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    def can_add_child(self):
        # 是否来可以添加新的子节点
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        # 当前节点是否胜负已分
        return self.game_state.is_over()

    def winning_frac(self, player):
        # 获取胜率
        return float(self.win_counts[player]) / float(self.num_rollouts)


class FastRandomBot(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.dim = None
        # 把所有可走步骤缓存起来
        self.point_cache = []

    def _update_cache(self, dim):
        # 如果换了棋盘，那么要重新扫描棋盘，重新获取可走步骤
        self.dim = dim
        rows, cols = dim
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                self.point_cache.append(Point(row=r, col=c))

    def select_move(self, game_state):
        dim = (game_state.board.num_rows, game_state.board.num_cols)
        if dim != self.dim:
            self._update_cache(dim)

        idx = np.arange(len(self.point_cache))
        np.random.shuffle(idx)
        # 每次选择落子步骤时不再重新扫描棋盘，而是从缓存中选取一种，从而加快速度
        for i in idx:
            p = self.point_cache[i]

            if game_state.is_valid_move(Move.play(p)) and not is_point_an_eye(game_state.board,
                                                                              p,
                                                                              game_state.next_player):
                return Move.play(p)

        return Move.resign()


class MCTSAgent(Agent):
    def __init__(self, num_rounds, temperature):
        Agent.__init__(self)
        # 限制模拟对弈的总次数
        self.num_rounds = num_rounds
        # exploitation-exploration 的分配比例
        self.temperature = temperature

    def select_move(self, game_state):
        '''
        在给定棋盘状况下，通过使用蒙特卡洛树搜索获得下一步走法
        '''
        # 根据当前棋盘状况设置根节点
        root = MCTSNode(game_state)

        for i in range(self.num_rounds):
            node = root
            # 如果当前节点已经展开了它所有子节点，那么从子节点中选择下一个要展开的节点
            while (not node.can_add_child()) and (not node.is_terminal()):
                # 根据选择公式，通过计算决定选择哪个子节点
                node = self.select_child(node)

            # 从当前节点对应的棋局中选择可走步骤形成下一个子节点
            if node.can_add_child():
                node = node.add_random_child()

            # 创建机器人，在展开节点对应棋盘基础上进行随机博弈
            print('before simulate in round: ', i)
            winner = self.simulate_random_game(node.game_state)
            print('after simulate in round: ', i)

            # 统计博弈结果，并将结果上传到所有父节点进行综合统计
            while node is not None:
                node.record_win(winner)
                node = node.parent

        # 在模拟完给定次数后，选择胜率最大的子节点对应的走法作为下一步棋的走法
        scored_moves = [
            (child.winning_frac(game_state.next_player), child.move, child.num_rollouts)
            for child in root.children
        ]
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        for s, m, n in scored_moves[:10]:
            print('%s - %.3f (%d)' % (m, s, n))

        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(game_state.next_player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move

        print('Select move %s with win pct %.3f' % (best_move, best_pct))
        return best_move

    def select_child(self, node):
        '''
        根据选择公式计算每个子节点的得分，然后选择得分最大的子节点
        '''
        N = node.num_rollouts
        log_N = math.log(N)

        best_score = -1
        best_child = None
        for child in node.children:
            w = child.winning_frac(node.game_state.next_player)
            n = child.num_rollouts
            score = w + self.temperature * math.sqrt(log_N / n)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    @staticmethod
    def simulate_random_game(game):
        # 启动两个机器人，在给定棋盘下随机落子
        bots = {
            Player.black: FastRandomBot(),
            Player.white: FastRandomBot()
        }

        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)

        return game.winner()


def print_move(player, move):
    if move.is_pass:
        move_str = 'passes'
    elif move.is_resign:
        move_str = 'resign'
    else:
        move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)
    print('%s %s' % (player, move_str))


def print_board(board):
    for row in range(board.num_rows, 0, -1):
        bump = ' ' if row <= 9 else ''
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%d %s' % (bump, row, ''.join(line)))

    print('   ' + ' '.join(COLS[:board.num_cols]))


def to_python(player_state):
    if player_state is None:
        return 'None'
    if player_state == Player.black:
        return Player.black
    return Player.white


# 把A3,D3这样的输入转换成具体坐标
def point_from_coords(coords):
    # 获取表示列的字母
    col = COLS.index(coords[0]) + 1
    # 获取表示行的数字
    row = int(coords[1:])
    return Point(row=row, col=col)


def print_bot_move(move):
    if move.is_pass:
        move_str = 'passes'
    elif move.is_resign:
        move_str = 'resign'
    else:
        move_str = (move.point.col - 1, move.point.row - 1)
    return move_str


# 棋盘的列用字母表示
COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: ' . ',
    Player.black: 'x',
    Player.white: 'o'
}

# 用一个64位整型对应每个棋盘
MAX63 = 0x7fffffffffffffff
# 3*19*19 / MAX63
# 发明这种编码算法的人叫zobrist
zobrist_HASH_CODE = {}

zobrist_EMPTY_BOARD = 0

for row in range(1, 20):
    for col in range(1, 20):
        for state in (None, Player.black, Player.white):
            # 随机选取一个整数对应当前位置,这里默认当前取随机值时不会与前面取值发生碰撞
            code = random.randint(0, MAX63)
            zobrist_HASH_CODE[Point(row, col), state] = code

print('HASH_CODE = {')
for (pt, state), hash_code in zobrist_HASH_CODE.items():
    print(' (%r, %s): %r,' % (pt, to_python(state), hash_code))

print('}')
print(' ')
print('EMPTY_BOARD = %d' % (zobrist_EMPTY_BOARD,))


def main():
    # 初始化9*9棋盘
    board_size = 5
    game = GameState.new_game(board_size)
    bot = MCTSAgent(100, temperature=10.0)

    while not game.is_over():
        print_board(game.board)
        if game.next_player == Player.black:
            human_move = input('-- ')
            point = point_from_coords(human_move.strip())

            move = Move.play(point)

        else:

            move = bot.select_move(game)
            bot_move = print_bot_move(move)

            print(bot_move[0], bot_move[1])

        print_move(game.next_player, move)
        game = game.apply_move(move)

    winner = game.winner()

    if winner == Player.black:
        print("the winner is black")

    elif winner == Player.white:
        print("the winner is white")


if __name__ == '__main__':
    main()

