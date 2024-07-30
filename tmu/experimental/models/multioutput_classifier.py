# Copyright (c) 2023 Ole-Christoffer Granmo
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from tmu.models.base import MultiWeightBankMixin, SingleClauseBankMixin, TMBaseModel
from tmu.util.encoded_data_cache import DataEncoderCache
from tmu.weight_bank import WeightBank


class TMCoalesceMultiOuputClassifier(TMBaseModel, SingleClauseBankMixin, MultiWeightBankMixin):
    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        q=5.0,
        type_i_ii_ratio=1.0,
        type_iii_feedback=False,
        focused_negative_sampling=False,
        output_balancing=False,
        d=200.0,
        platform="CPU",
        patch_dim=None,
        feature_negation=True,
        boost_true_positive_feedback=1,
        reuse_random_feedback=0,
        max_positive_clauses=None,
        max_included_literals=None,
        number_of_state_bits_ta=8,
        number_of_state_bits_ind=8,
        weighted_clauses=False,
        clause_drop_p=0.0,
        literal_drop_p=0.0,
        seed=None,
    ):
        super().__init__(
            number_of_clauses=number_of_clauses,
            T=T,
            s=s,
            type_i_ii_ratio=type_i_ii_ratio,
            type_iii_feedback=type_iii_feedback,
            focused_negative_sampling=focused_negative_sampling,
            output_balancing=output_balancing,
            d=d,
            platform=platform,
            patch_dim=patch_dim,
            feature_negation=feature_negation,
            boost_true_positive_feedback=boost_true_positive_feedback,
            reuse_random_feedback=reuse_random_feedback,
            max_included_literals=max_included_literals,
            number_of_state_bits_ta=number_of_state_bits_ta,
            number_of_state_bits_ind=number_of_state_bits_ind,
            weighted_clauses=weighted_clauses,
            clause_drop_p=clause_drop_p,
            literal_drop_p=literal_drop_p,
            seed=seed,
        )
        SingleClauseBankMixin.__init__(self)
        MultiWeightBankMixin.__init__(self, seed=seed)

        # These data structures cache the encoded data for the training and test sets. It also makes a fast-check if
        # training data has changed, and only re-encodes if it has.
        self.test_encoder_cache = DataEncoderCache(seed=self.seed)
        self.train_encoder_cache = DataEncoderCache(seed=self.seed)

        self.max_positive_clauses = max_positive_clauses
        self.q = q

    def init_clause_bank(self, X: np.ndarray, Y: np.ndarray):
        clause_bank_type, clause_bank_args = self.build_clause_bank(X=X)
        self.clause_bank = clause_bank_type(**clause_bank_args)

    def init_weight_bank(self, X: np.ndarray, Y: np.ndarray):
        self.number_of_classes = Y.shape[1]
        self.weight_banks.set_clause_init(
            WeightBank,
            dict(weights=self.rng.choice([-1, 1], size=self.number_of_clauses).astype(np.int32)),
        )
        self.weight_banks.populate(list(range(self.number_of_classes)))

    def init_after(self, X: np.ndarray, Y: np.ndarray):
        if self.max_included_literals is None:
            self.max_included_literals = self.clause_bank.number_of_literals

        if self.max_positive_clauses is None:
            self.max_positive_clauses = self.number_of_clauses

    def fit(self, X, Y, shuffle=True, progress_bar=False, **kwargs):
        self.init(X, Y)

        encoded_X_train = self.train_encoder_cache.get_encoded_data(X, encoder_func=lambda x: self.clause_bank.prepare_X(X))
        Y_csr = csr_matrix(Y)

        # Drops clauses randomly based on clause drop probability
        self.clause_active = (self.rng.rand(self.number_of_clauses) >= self.clause_drop_p).astype(np.int32)

        # Literals are dropped based on literal drop probability
        self.literal_active = np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32)
        literal_active_integer = self.rng.rand(self.clause_bank.number_of_literals) >= self.literal_drop_p
        for k in range(self.clause_bank.number_of_literals):
            if literal_active_integer[k] == 1:
                ta_chunk = k // 32
                chunk_pos = k % 32
                self.literal_active[ta_chunk] |= 1 << chunk_pos

        if not self.feature_negation:
            for k in range(
                self.clause_bank.number_of_literals // 2,
                self.clause_bank.number_of_literals,
            ):
                ta_chunk = k // 32
                chunk_pos = k % 32
                self.literal_active[ta_chunk] &= ~(1 << chunk_pos)

        self.literal_active = self.literal_active.astype(np.uint32)

        shuffled_index = np.arange(X.shape[0])
        if shuffle:
            self.rng.shuffle(shuffled_index)

        pbar = tqdm(shuffled_index) if progress_bar else shuffled_index

        # Combine all weight banks, to make use of faster numpy matrix operation
        self.wcomb = np.empty((self.number_of_clauses, self.number_of_classes), dtype=np.int32)
        for c in range(self.number_of_classes):
            self.wcomb[:, c] = self.weight_banks[c].get_weights()

        for e in pbar:
            y_csr = Y_csr[e, :]
            pos_class_ind = y_csr.indices

            clause_outputs = self.clause_bank.calculate_clause_outputs_update(self.literal_active, encoded_X_train, e)

            class_sums = (clause_outputs * self.clause_active)[np.newaxis, :] @ self.wcomb
            class_sums = np.clip(class_sums, -self.T, self.T).astype(np.int32).ravel()

            t = -self.T * np.ones(self.number_of_classes)
            t[pos_class_ind] *= -1

            self.update_ps = (t - class_sums) / (2 * t)

            if self.update_ps.sum() == 0:
                continue

            for c in range(self.number_of_classes):
                update_p = self.update_ps[c]
                if Y[e, c] == 1:
                    self.clause_bank.type_i_feedback(
                        update_p=update_p * self.type_i_p,
                        clause_active=self.clause_active * (self.weight_banks[c].get_weights() >= 0),
                        literal_active=self.literal_active,
                        encoded_X=encoded_X_train,
                        e=e,
                    )
                    self.clause_bank.type_ii_feedback(
                        update_p=update_p * self.type_ii_p,
                        clause_active=self.clause_active * (self.weight_banks[c].get_weights() < 0),
                        literal_active=self.literal_active,
                        encoded_X=encoded_X_train,
                        e=e,
                    )
                    if (self.weight_banks[c].get_weights() >= 0).sum() < self.max_positive_clauses:
                        self.weight_banks[c].increment(
                            clause_output=clause_outputs,
                            update_p=update_p,
                            clause_active=self.clause_active,
                            positive_weights=True,
                        )
                        self.wcomb[:, c] = self.weight_banks[c].get_weights()

                    self.update_ps[c] = 0.0

                else:
                    if update_p and self.rng.uniform() <= (self.q / max(1, self.number_of_classes - 1)):
                        self.clause_bank.type_i_feedback(
                            update_p=update_p * self.type_i_p,
                            clause_active=self.clause_active * (self.weight_banks[c].get_weights() < 0),
                            literal_active=self.literal_active,
                            encoded_X=encoded_X_train,
                            e=e,
                        )

                        self.clause_bank.type_ii_feedback(
                            update_p=update_p * self.type_ii_p,
                            clause_active=self.clause_active * (self.weight_banks[c].get_weights() >= 0),
                            literal_active=self.literal_active,
                            encoded_X=encoded_X_train,
                            e=e,
                        )

                        self.weight_banks[c].decrement(
                            clause_output=clause_outputs,
                            update_p=update_p,
                            clause_active=self.clause_active,
                            negative_weights=True,
                        )
                        self.wcomb[:, c] = self.weight_banks[c].get_weights()
                        self.update_ps[c] = 0.0

            # for c in pos_class_ind:
            #     self.clause_bank.type_i_feedback(
            #         update_p=self.update_ps[c] * self.type_i_p,
            #         clause_active=self.clause_active
            #         * (self.weight_banks[c].get_weights() >= 0),
            #         literal_active=self.literal_active,
            #         encoded_X=encoded_X_train,
            #         e=e,
            #     )
            #     self.clause_bank.type_ii_feedback(
            #         update_p=self.update_ps[c] * self.type_ii_p,
            #         clause_active=self.clause_active
            #         * (self.weight_banks[c].get_weights() < 0),
            #         literal_active=self.literal_active,
            #         encoded_X=encoded_X_train,
            #         e=e,
            #     )
            #     if (
            #         self.weight_banks[c].get_weights() >= 0
            #     ).sum() < self.max_positive_clauses:
            #         self.weight_banks[c].increment(
            #             clause_output=clause_outputs,
            #             update_p=self.update_ps[c],
            #             clause_active=self.clause_active,
            #             positive_weights=True,
            #         )
            #         self.wcomb[:, c] = self.weight_banks[c].get_weights()
            #
            #     self.update_ps[c] = 0.0
            #
            # if self.update_ps.sum() == 0:
            #     continue

            # if self.focused_negative_sampling:
            #     not_target = self.rng.choice(
            #         self.number_of_classes, p=self.update_ps / self.update_ps.sum()
            #     )
            #     update_p = self.update_ps[not_target]
            # else:
            #     not_target = self.rng.choice(np.argwhere(self.update_ps != 0).flatten())
            #     update_p = self.update_ps[not_target]
            #     assert (
            #         update_p != 0.0
            #     ), "myassert: neg_classes not working, update_p became 0."
            #
            # self.clause_bank.type_i_feedback(
            #     update_p=update_p * self.type_i_p,
            #     clause_active=self.clause_active
            #     * (self.weight_banks[not_target].get_weights() < 0),
            #     literal_active=self.literal_active,
            #     encoded_X=encoded_X_train,
            #     e=e,
            # )
            #
            # self.clause_bank.type_ii_feedback(
            #     update_p=update_p * self.type_ii_p,
            #     clause_active=self.clause_active
            #     * (self.weight_banks[not_target].get_weights() >= 0),
            #     literal_active=self.literal_active,
            #     encoded_X=encoded_X_train,
            #     e=e,
            # )
            #
            # self.weight_banks[not_target].decrement(
            #     clause_output=clause_outputs,
            #     update_p=update_p,
            #     clause_active=self.clause_active,
            #     negative_weights=True,
            # )
            # self.wcomb[:, not_target] = self.weight_banks[not_target].get_weights()

        return

    def predict(
        self,
        X,
        shuffle=False,
        clip_class_sum=False,
        return_class_sums: bool = False,
        progress_bar=False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        encoded_X_test = self.clause_bank.prepare_X(X)

        for c in range(self.number_of_classes):
            self.wcomb[:, c] = self.weight_banks[c].get_weights()

        shuffled_index = np.arange(X.shape[0])
        if shuffle:
            self.rng.shuffle(shuffled_index)
        pbar = tqdm(shuffled_index) if progress_bar else shuffled_index

        # Compute class sums for all samples
        class_sums = np.empty((X.shape[0], self.number_of_classes))
        for e in pbar:
            class_sums[e, :] = self.compute_class_sums(encoded_X_test, e, clip_class_sum)

        output = (class_sums >= 0).astype(np.uint32)

        if return_class_sums:
            return output, class_sums
        else:
            return output

    def compute_class_sums(self, encoded_X_test, ith_sample: int, clip_class_sum: bool):
        """The following function evaluates the resulting class sum votes.

        Args:
            ith_sample (int): The index of the sample
            clip_class_sum (bool): Wether to clip class sums

        Returns:
            list[int]: list of all class sums
        """
        clause_outputs = self.clause_bank.calculate_clause_outputs_predict(encoded_X_test, ith_sample)
        class_sums = clause_outputs[np.newaxis, :] @ self.wcomb
        if clip_class_sum:
            class_sums = np.clip(class_sums, -self.T, self.T).astype(np.int32)
        return class_sums

    def to_cpu(self, X):
        if self.platform in ["GPU", "CUDA"]:
            clause_bank_gpu = self.clause_bank
            clause_bank_gpu.synchronize_clause_bank()
            clause_bank_type, clause_bank_args = self._build_cpu_bank(X)
            clause_bank_cpu = clause_bank_type(**clause_bank_args)

            clause_bank_cpu.clause_bank = clause_bank_gpu.clause_bank
            clause_bank_cpu.clause_output = clause_bank_gpu.clause_output
            clause_bank_cpu.literal_clause_count = clause_bank_gpu.literal_clause_count

            clause_bank_cpu._cffi_init()

            self.clause_bank = clause_bank_cpu
            self.platform = "CPU"
            print("to_cpu(): Successful....")

        elif self.platform == "CPU":
            print("to_cpu(): Already CPU....")

        else:
            print("to_cpu(): Not implemented....")

    def clause_precision(self, the_class, X, Y):
        clause_outputs = self.transform(X)
        weights = self.weight_banks[the_class].get_weights()

        positive_clause_outputs = (weights >= 0)[:, np.newaxis].transpose() * clause_outputs
        true_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(axis=0)
        false_positive_clause_outputs = positive_clause_outputs[Y != the_class].sum(axis=0)

        positive_clause_outputs = (weights < 0)[:, np.newaxis].transpose() * clause_outputs
        true_positive_clause_outputs += positive_clause_outputs[Y != the_class].sum(axis=0)
        false_positive_clause_outputs += positive_clause_outputs[Y == the_class].sum(axis=0)

        return np.where(
            true_positive_clause_outputs + false_positive_clause_outputs == 0,
            0,
            1.0 * true_positive_clause_outputs / (true_positive_clause_outputs + false_positive_clause_outputs),
        )

    def clause_recall(self, the_class, X, Y):
        clause_outputs = self.transform(X)
        weights = self.weight_banks[the_class].get_weights()

        positive_clause_outputs = (weights >= 0)[:, np.newaxis].transpose() * clause_outputs
        true_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(axis=0) / Y[Y == the_class].shape[0]

        positive_clause_outputs = (weights < 0)[:, np.newaxis].transpose() * clause_outputs
        true_positive_clause_outputs += positive_clause_outputs[Y != the_class].sum(axis=0) / Y[Y != the_class].shape[0]

        return true_positive_clause_outputs

    def get_weights(self, the_class):
        return self.weight_banks[the_class].get_weights()

    def get_weight(self, the_class, clause):
        return self.weight_banks[the_class].get_weights()[clause]

    def set_weight(self, the_class, clause, weight):
        self.weight_banks[the_class].get_weights()[clause] = weight

    def number_of_include_actions(self, clause):
        return self.clause_bank.number_of_include_actions(clause)
