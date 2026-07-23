from unittest.mock import Mock, patch

from langchain_core.documents import Document

from digital_twin.vector_store import create_or_get_vector_store


def test_create_or_get_vector_store_populates_empty_store() -> None:
    chunks = [
        Document(page_content="First chunk"),
        Document(page_content="Second chunk"),
    ]
    embeddings = Mock()

    mock_vector_store = Mock()
    mock_vector_store.get.return_value = {"ids": ["1", "2"]}
    mock_vector_store._collection.count.return_value = 0

    with patch(
        "digital_twin.vector_store.Chroma",
        return_value=mock_vector_store,
    ) as mock_chroma:
        result = create_or_get_vector_store(
            embedding=embeddings,
            chunks=chunks,
            chroma_collection="test-collection",
            chroma_directory="persistence/chroma_db_test",
            reset_vector_store=False,
        )

    mock_chroma.assert_called_once_with(
        collection_name="test-collection",
        embedding_function=embeddings,
        persist_directory="persistence/chroma_db_test",
    )
    mock_vector_store.add_documents.assert_called_once_with(chunks)
    mock_vector_store._collection.count.assert_called_once_with()
    mock_vector_store.reset_collection.assert_not_called()
    assert mock_vector_store._collection.count() == 0
    assert result is mock_vector_store


def test_create_or_get_vector_store_does_not_add_new_documents_unless_requested() -> (
    None
):
    chunks = [Document(page_content="New chunk")]
    embeddings = Mock()

    mock_vector_store = Mock()
    mock_vector_store._collection.count.return_value = 5
    mock_vector_store.get.return_value = {"ids": ["1", "2", "3", "4", "5"]}

    with patch(
        "digital_twin.vector_store.Chroma",
        return_value=mock_vector_store,
    ):
        result = create_or_get_vector_store(
            embedding=embeddings,
            chunks=chunks,
            chroma_collection="test-collection",
            chroma_directory=str("persistence/chroma_db_test"),
            reset_vector_store=False,
        )

    mock_vector_store._collection.count.assert_called_once_with()
    mock_vector_store.add_documents.assert_not_called()
    assert result is mock_vector_store


def test_create_or_get_vector_store_resets_and_populates() -> None:
    chunks = [Document(page_content="Replacement chunk")]
    embeddings = Mock()

    mock_vector_store = Mock()
    mock_vector_store._collection.count.return_value = 0
    mock_vector_store.get.return_value = {"ids": ["1"]}

    with patch(
        "digital_twin.vector_store.Chroma",
        return_value=mock_vector_store,
    ):
        create_or_get_vector_store(
            embedding=embeddings,
            chunks=chunks,
            chroma_collection="test-collection",
            chroma_directory=str("persistence/chroma_db_test"),
            reset_vector_store=True,
        )

    mock_vector_store.reset_collection.assert_called_once_with()
    mock_vector_store._collection.count.assert_called_once_with()
    mock_vector_store.add_documents.assert_called_once_with(chunks)
