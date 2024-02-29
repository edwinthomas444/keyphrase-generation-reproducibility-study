import torch
import torch.nn as nn

from onmt.utils.misc import aeq
from onmt.utils.loss import CommonLossCompute


def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs=None,
                         batch_dim=1, batch_offset=None):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset = len(tgt_vocab)  # vocab_size
    for b in range(scores.size(batch_dim)):
        blank = []
        fill = []

        if src_vocabs is None:
            src_vocab = batch.src_ex_vocab[b]
        else:
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            src_vocab = src_vocabs[index]

        for i in range(1, len(src_vocab)):
            sw = src_vocab.itos[i]
            #  (deprecated) normalize subwords to reduce mismatches, but it causes many unmergable words such as '2-deg enerATE' ['Ġ2', '-', 'deg', 'Ġener', 'ATE']
            # if hasattr(tgt_vocab, 'norm_stoi'):
            #     ti = tgt_vocab.norm_stoi[sw]
            # else:
            #     ti = tgt_vocab.stoi[sw]
            ti = tgt_vocab.stoi[sw]

            # @memray, for pretrained tokenizer, pad_index can be non-zero
            pad_index = tgt_vocab.pad_index if hasattr(tgt_vocab, 'pad_index') else 0
            if ti != pad_index:
                blank.append(offset + i)
                fill.append(ti)
        if blank:
            blank = torch.Tensor(blank).type_as(batch.indices.data)
            fill = torch.Tensor(fill).type_as(batch.indices.data)
            score = scores[:, b] if batch_dim == 1 else scores[b]
            score.index_add_(1, fill, score.index_select(1, blank))
            score.index_fill_(1, blank, 1e-10)
    return scores


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.

    These networks consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size, output_size, pad_idx):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)``
        """

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


class CopyGeneratorLoss(nn.Module):
    """Copy generator criterion."""
    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, align, target):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            align (LongTensor): ``(batch_size x tgt_len)``
            target (LongTensor): ``(batch_size x tgt_len)``
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss


class CommonCopyGeneratorLossCompute(CommonLossCompute):
    """Common Copy Generator Loss Computation."""
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length,
                 lambda_coverage=0.0, lambda_align=0.0, tgt_shift_index=1,
                 lambda_orth_reg=0.0, lambda_sem_cov=0.0,
                 n_neg=32, semcov_ending_state=False,
                 sep_idx=None, eos_idx=None,
                 **kwargs):
        super(CommonCopyGeneratorLossCompute, self).__init__(
            criterion, generator, lambda_coverage, lambda_align, tgt_shift_index, **kwargs
        )
        self.sep_idx = sep_idx
        self.eos_idx = eos_idx
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length
        self.lambda_coverage = lambda_coverage
        self.tgt_shift_index = tgt_shift_index
        self.lambda_orth_reg = lambda_orth_reg
        self.lambda_sem_cov = lambda_sem_cov
        self.n_neg = n_neg
        self.semcov_ending_state = semcov_ending_state

    def _compute_loss(self, batch, output, target, copy_attn, align,
                      std_attn=None, coverage_attn=None,
                      src_states=None, dec_states=None, tgtenc_states=None,
                      model=None
                      ):
        """Compute the loss.

        The args must match :func:`self._make_shard_state()`.

        Args:
            batch: the current batch.
            output: the predict output from the model, shape=[T-1, B, H].
            target: the validate target to compare output with, shape=[T-1, B, S].
            copy_attn: the copy attention value, shape=[T-1, B, S].
            align: the align info, shape=[T-1, B].
        """
        target_indices = target # before flattening

        target = target.contiguous().view(-1) # [T-1, B] -> (T-1)*B
        align = align.view(-1) # (T-1)*B

        scores = self.generator( # [(T-1)*B, V+tmp_vocab_size]
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )

        loss = self.criterion(scores, align, target) # (T-1)*B
        # print("loss=%.5f" % loss.mean().item())

        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(std_attn,
                                                        coverage_attn)
            loss += coverage_loss

        # compute orthogonal penalty loss
        if self.lambda_orth_reg > 0.0:
            assert dec_states is not None
            # decoder hidden state: output of decoder
            orthogonal_penalty = self._compute_orthogonal_regularization_loss(target_indices, dec_states, self.sep_idx)
            loss += orthogonal_penalty
            # print("Orth_reg=%.5f" % orthogonal_penalty)

        # compute semantic coverage loss for target encoder
        if self.lambda_sem_cov > 0.0:
            assert model is not None
            assert src_states is not None
            assert tgtenc_states is not None
            semantic_coverage_loss = self._compute_semantic_coverage_loss(model,
                                                                          src_states, tgtenc_states,
                                                                          target_indices,
                                                                          num_negative=self.n_neg,
                                                                          semcov_ending_state=self.semcov_ending_state,
                                                                          sep_idx=self.sep_idx, eos_idx=self.eos_idx,
                                                                          )
            loss += semantic_coverage_loss
            # print("Sem_cov=%.5f\n" % semantic_coverage_loss)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, None)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # Compute sum of perplexities for stats
        stats = self._stats(loss.sum().clone(), scores_data, target_data, batch_size=batch.batch_size)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats

    def _make_shard_state(self, batch, output, range_, attns):
        """See base class for args description."""
        shard_state = super(CommonCopyGeneratorLossCompute,
                            self)._make_shard_state(batch, output,
                                                    range_, attns)

        start_range = range_[0] + self.tgt_shift_index
        end_range = range_[1]
        shard_state.update({
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[start_range: end_range]
        })
        return shard_state


class CopyGeneratorLossCompute(CommonCopyGeneratorLossCompute):
    """Copy Generator Loss Computation."""
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length, **kwargs):
        super(CopyGeneratorLossCompute, self).__init__(criterion, generator,
                                                       tgt_vocab,
                                                       normalize_by_length,
                                                       # lambda_coverage=0.0, # @memray
                                                       # lambda_align=0.0,
                                                       tgt_shift_index=1,
                                                       **kwargs)


class CopyGeneratorLMLossCompute(CommonCopyGeneratorLossCompute):
    """Copy Generator LM Loss Computation."""
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length,
                 lambda_coverage=0.0):
        super(CopyGeneratorLMLossCompute, self).__init__(criterion, generator,
                                                         tgt_vocab,
                                                         normalize_by_length,
                                                         lambda_coverage=0.0,
                                                         tgt_shift_index=0)
