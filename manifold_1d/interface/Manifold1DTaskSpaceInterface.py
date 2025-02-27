import copy
import numpy as np
from .ReferencePosition import ReferencePosition
from robel_dclaw_kinematics import ForwardKinematics
from ..utils import D_to_NTD
from ..utils import create_cyclic_data
from ...utils import AbstractTaskSpaceInterface


class Manifold1DTaskSpaceInterface(AbstractTaskSpaceInterface):
    def __init__(self):
        self.min                             = 0.0
        self.max                             = 1.0
        self.forward_kinematics              = ForwardKinematics()
        self.reference_joint_position        = ReferencePosition().augument_reference_joint_position()
        self.reference_end_effector_position = self.forward_kinematics.calc_1claw(self.reference_joint_position)
        self.reference_end_effector_position = create_cyclic_data(self.reference_end_effector_position) # 中間点を補完する際にはtask_spaceとして閉じている必要がある
        self.reference_task_space_position   = self._create_reference_task_space_position()
        self.num_claw                        = 3


    def _create_reference_task_space_position(self):
        diff_reference_end_effector                     = np.diff(self.reference_end_effector_position, n=1, axis=0)          # xyzの各次元での差分
        euclidean_distance_for_each_reference           = np.sqrt(np.sum(diff_reference_end_effector**2, axis=-1))            # 差分を2乗してxyzの次元を総和して平方根を取ることで各点間のユークリッド距離を計算
        cumulative_euclidean_distance                   = np.cumsum(euclidean_distance_for_each_reference, axis=0)            # ユークリッド距離の累積和を計算
        self.max_euclidean_distance                     = cumulative_euclidean_distance.max()                                 # ユークリッド距離の累積和の最大値を取得
        normalized_cumulative_euclidean_distance        = cumulative_euclidean_distance / self.max_euclidean_distance         # ユークリッド距離の累積和を最大値で割ることで[0, 1]の範囲に正規化
        cyclic_normalized_cumulative_euclidean_distance = np.hstack((np.zeros(1), normalized_cumulative_euclidean_distance))  # 最初の点までをつなぐ開始点として0を追加
        return cyclic_normalized_cumulative_euclidean_distance


    def task2end(self, task_space_position) -> np.ndarray:
        task_space_position = D_to_NTD(task_space_position)
        sequence, step, dim = task_space_position.shape
        assert dim == 3 # 指1本あたり1次元に拘束するので3本で3次元
        end_effector_list  = [0] * dim
        for d in range(dim):
            end_effector_list[d] = self._get_end_effector_position_from_task_space_position(task_space_position[:,:,d].reshape(-1))
        flatten_end_effector_position = np.concatenate(end_effector_list, axis=-1)
        return flatten_end_effector_position


    def _get_end_effector_position_from_task_space_position(self, task_space_position):
        '''
        - 中間点を補完する際に，単純にreference_task_space_positionとの差だけで補完に使用する2点を決定してしまうと
        必ず補完の空洞になる領域が発生してしまい，連続的な行動空間を生成できない（点同士が距離が離れている領域付近で欠落する）
        - なのでnp.argsort(dist_matrix)から上位2つの近傍点を取得するだけでは正しく補完できない
        - 理由：reference_task_space_positionにある各点同士の距離は一定ではないため
        '''
        assert len(task_space_position.shape) == 1                                                                                                      # 1本あたりの計算であることを確認
        clipped_task_space_position        = task_space_position % self.max                                                                             # 値をtask_spaceの範囲内にクリップ
        distance_matrix                    = clipped_task_space_position.reshape(-1, 1) - self.reference_task_space_position.reshape(1, -1)             # referenceとの差を計算
        signed_distance_matrix             = np.sign(distance_matrix)                                                                                   # referenceとの差をの符号を取得
        signed_distance_matrix             = self._convet_zero_to_plus1_or_minus1_for_calculating_difference_of_sign(signed_distance_matrix)            # 0があると後の計算で困るので1か-1に変換しておく
        index_sign_changeed                = np.nonzero(np.diff(signed_distance_matrix, n=1, axis=-1))[1]                                               # referenceとの差で符号関係が変化する点を探す（以下，以上の関係が変化する点）
        index_top2_nearest_neighbor        = np.concatenate((index_sign_changeed.reshape(-1, 1), index_sign_changeed.reshape(-1, 1)+1), axis=-1)        # 符号関係が変化する位置から補完に用いる2点を取得
        top2_nearest_task_space_position   = np.take(self.reference_task_space_position, index_top2_nearest_neighbor)                                   # 補完に用いるtask_space_space_positionを取得
        top2_nearest_end_effector_position = np.take(self.reference_end_effector_position, index_top2_nearest_neighbor, axis=0)                         # 補完に用いるend_effector_positionを取得
        direction_vector                   = np.diff(top2_nearest_end_effector_position, n=1, axis=1)                                                   # 補完に用いる方向ベクトルを計算
        direction_vector_squeezed          = np.squeeze(direction_vector, axis=1)                                                                       # データ形状を整形
        unit_direction_vector              = direction_vector_squeezed / np.linalg.norm(direction_vector_squeezed, axis=-1, keepdims=True)              # 方向ベクトルと同一方向の単位ベクトルを計算
        t                                  = np.abs(clipped_task_space_position - top2_nearest_task_space_position[:,0]) * self.max_euclidean_distance  # task_space_positionを表すための媒介変を計算
        end_effector_position              = top2_nearest_end_effector_position[:, 0] + t.reshape(-1, 1) * unit_direction_vector                        # 補完点を計算
        return end_effector_position


    def _convet_zero_to_plus1_or_minus1_for_calculating_difference_of_sign(self, input_array):
        # edge index preprocesssing
        input_array_inserted_minus1_at_index0 = self._insert_specific_value_instead_of_zero_dependig_on_the_index(input_array, value=1, index=0)
        input_array_inserted_minus1_and_plus1 = self._insert_specific_value_instead_of_zero_dependig_on_the_index(input_array_inserted_minus1_at_index0, value=-1, index=-1)
        # non-edge index preprocesssing
        index_zero_elements = np.where(input_array_inserted_minus1_and_plus1==0)
        input_array_inserted_minus1_and_plus1[index_zero_elements] = 1
        return input_array_inserted_minus1_and_plus1


    def _insert_specific_value_instead_of_zero_dependig_on_the_index(self, input_array, value, index):
        input_array = copy.deepcopy(input_array)
        assert (len(input_array.shape) == 2) and (input_array.shape[1] == self.reference_task_space_position.shape[0])
        zero_element_index = np.where(input_array[:, index]==0)[0]
        input_array[zero_element_index, index] = value
        return input_array


    def end2task(self, end_effector_position):
        # import ipdb; ipdb.set_trace()
        end_effector_position = D_to_NTD(end_effector_position)
        sequence, step, dim   = end_effector_position.shape
        assert dim == 9 # 指1本あたり3次元なので3本で9次元
        task_space_position = np.zeros(self.num_claw)
        for i, _end_effector_position in enumerate(np.split(end_effector_position, self.num_claw, axis=-1)):
            task_space_position[i] = self._get_task_space_position_from_end_effector_position(_end_effector_position[0])
        task_space_position = D_to_NTD(task_space_position)
        return task_space_position


    def _get_task_space_position_from_end_effector_position(self, end_effector_position):
        assert end_effector_position.shape == (1, 3) # １つのendeffector座標を比較するため
        distance               = np.linalg.norm(self.reference_end_effector_position - end_effector_position, axis=-1)
        index_minimum_distance = np.argsort(distance)[0]
        nearest_reference      = self.reference_task_space_position[index_minimum_distance]
        return nearest_reference

