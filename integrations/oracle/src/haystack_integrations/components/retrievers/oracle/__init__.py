# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from .embedding_retriever import OracleEmbeddingRetriever
from .sparse_embedding_retriever import OracleSparseEmbeddingRetriever

__all__ = ["OracleEmbeddingRetriever", "OracleSparseEmbeddingRetriever"]
