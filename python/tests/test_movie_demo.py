"""Tests for the movie demo script."""

import csv
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from movie_demo import (
    Movie,
    load_movies_from_csv,
    generate_sample_movies,
    load_and_embed_movies,
    search_similar_movies,
    interactive_mode,
)


class TestMovie:
    """Tests for the Movie dataclass."""

    def test_movie_creation(self):
        """Test creating a Movie instance."""
        movie = Movie(
            title="Test Movie",
            year="2024",
            genre="Action, Adventure",
            overview="A test movie overview.",
            director="Test Director",
            rating="8.5"
        )
        assert movie.title == "Test Movie"
        assert movie.year == "2024"
        assert movie.genre == "Action, Adventure"
        assert movie.overview == "A test movie overview."
        assert movie.director == "Test Director"
        assert movie.rating == "8.5"

    def test_movie_to_text(self):
        """Test converting movie to text for embedding."""
        movie = Movie(
            title="The Matrix",
            year="1999",
            genre="Sci-Fi",
            overview="A hacker discovers reality.",
            director="Wachowskis",
            rating="8.7"
        )
        text = movie.to_text()
        assert "The Matrix" in text
        assert "A hacker discovers reality." in text
        assert "Genre: Sci-Fi" in text

    def test_movie_to_text_minimal(self):
        """Test converting movie with minimal data to text."""
        movie = Movie(
            title="Simple Movie",
            year="",
            genre="",
            overview="",
            director="",
            rating=""
        )
        text = movie.to_text()
        assert text == "Simple Movie"


class TestLoadMoviesFromCSV:
    """Tests for CSV loading functionality."""

    def test_load_movies_standard_format(self):
        """Test loading movies from standard IMDB format CSV."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Series_Title', 'Released_Year', 'Genre', 'Overview', 'Director', 'IMDB_Rating'])
            writer.writerow(['The Shawshank Redemption', '1994', 'Drama', 'Two imprisoned men bond.', 'Frank Darabont', '9.3'])
            writer.writerow(['The Godfather', '1972', 'Crime, Drama', 'An aging patriarch.', 'Francis Ford Coppola', '9.2'])
            f.flush()
            filepath = f.name

        try:
            movies = load_movies_from_csv(filepath)
            assert len(movies) == 2
            assert movies[0].title == "The Shawshank Redemption"
            assert movies[0].year == "1994"
            assert movies[0].genre == "Drama"
            assert movies[0].rating == "9.3"
            assert movies[1].title == "The Godfather"
        finally:
            os.unlink(filepath)

    def test_load_movies_alternate_column_names(self):
        """Test loading movies with alternate column names."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Title', 'Year', 'Genres', 'Description', 'Directors', 'Score'])
            writer.writerow(['Test Movie', '2024', 'Action', 'A test overview.', 'Test Director', '8.0'])
            f.flush()
            filepath = f.name

        try:
            movies = load_movies_from_csv(filepath)
            assert len(movies) == 1
            assert movies[0].title == "Test Movie"
            assert movies[0].year == "2024"
            assert movies[0].genre == "Action"
        finally:
            os.unlink(filepath)

    def test_load_movies_missing_optional_columns(self):
        """Test loading movies when optional columns are missing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Title'])
            writer.writerow(['Minimal Movie'])
            f.flush()
            filepath = f.name

        try:
            movies = load_movies_from_csv(filepath)
            assert len(movies) == 1
            assert movies[0].title == "Minimal Movie"
            assert movies[0].year == ""
            assert movies[0].genre == ""
        finally:
            os.unlink(filepath)

    def test_load_movies_skips_empty_titles(self):
        """Test that movies with empty titles are skipped."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Title', 'Year'])
            writer.writerow(['Valid Movie', '2024'])
            writer.writerow(['', '2023'])  # Empty title
            writer.writerow(['Another Valid', '2022'])
            f.flush()
            filepath = f.name

        try:
            movies = load_movies_from_csv(filepath)
            assert len(movies) == 2
            assert movies[0].title == "Valid Movie"
            assert movies[1].title == "Another Valid"
        finally:
            os.unlink(filepath)

    def test_load_movies_missing_title_column_raises(self):
        """Test that missing title column raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Year', 'Genre'])
            writer.writerow(['2024', 'Action'])
            f.flush()
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="Could not find title column"):
                load_movies_from_csv(filepath)
        finally:
            os.unlink(filepath)


class TestGenerateSampleMovies:
    """Tests for sample movie generation."""

    def test_generate_sample_movies_returns_list(self):
        """Test that generate_sample_movies returns a list of movies."""
        movies = generate_sample_movies()
        assert isinstance(movies, list)
        assert len(movies) > 0

    def test_generate_sample_movies_has_valid_movies(self):
        """Test that sample movies have required fields."""
        movies = generate_sample_movies()
        for movie in movies:
            assert isinstance(movie, Movie)
            assert movie.title
            assert movie.year
            assert movie.genre
            assert movie.overview

    def test_generate_sample_movies_count(self):
        """Test that sample movies returns expected count."""
        movies = generate_sample_movies()
        # Should have 20 sample movies
        assert len(movies) == 20


class TestLoadAndEmbedMovies:
    """Tests for the load and embed functionality."""

    def test_load_and_embed_movies_success(self):
        """Test loading and embedding movies into Vector++."""
        # Mock client
        mock_client = Mock()
        mock_client.insert.side_effect = [f"uuid-{i}" for i in range(3)]

        # Mock embeddings
        mock_embeddings = Mock()
        mock_embeddings.model_name = "test-model"
        mock_embeddings.dimensions = 384
        mock_embeddings.embed_batch.return_value = [
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384,
        ]

        movies = [
            Movie("Movie 1", "2020", "Action", "Overview 1", "Director 1", "8.0"),
            Movie("Movie 2", "2021", "Drama", "Overview 2", "Director 2", "7.5"),
            Movie("Movie 3", "2022", "Comedy", "Overview 3", "Director 3", "7.0"),
        ]

        id_to_movie = load_and_embed_movies(mock_client, movies, mock_embeddings, batch_size=10)

        assert len(id_to_movie) == 3
        assert "uuid-0" in id_to_movie
        assert "uuid-1" in id_to_movie
        assert "uuid-2" in id_to_movie
        assert id_to_movie["uuid-0"].title == "Movie 1"

    def test_load_and_embed_movies_handles_errors(self):
        """Test that embedding errors are handled gracefully."""
        from vectorpp import VectorPPError

        mock_client = Mock()
        # First insert succeeds, second fails
        mock_client.insert.side_effect = ["uuid-0", VectorPPError("Test error"), "uuid-2"]

        mock_embeddings = Mock()
        mock_embeddings.model_name = "test-model"
        mock_embeddings.dimensions = 384
        mock_embeddings.embed_batch.return_value = [
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384,
        ]

        movies = [
            Movie("Movie 1", "2020", "Action", "Overview 1", "Director 1", "8.0"),
            Movie("Movie 2", "2021", "Drama", "Overview 2", "Director 2", "7.5"),
            Movie("Movie 3", "2022", "Comedy", "Overview 3", "Director 3", "7.0"),
        ]

        id_to_movie = load_and_embed_movies(mock_client, movies, mock_embeddings, batch_size=10)

        # Should have loaded 2 movies (one failed)
        assert len(id_to_movie) == 2


class TestSearchSimilarMovies:
    """Tests for search functionality."""

    def test_search_similar_movies_success(self, capsys):
        """Test searching for similar movies."""
        from vectorpp import SearchResult

        mock_client = Mock()
        mock_client.search.return_value = [
            SearchResult(id="uuid-0", score=0.95, metadata="Action"),
            SearchResult(id="uuid-1", score=0.85, metadata="Drama"),
        ]

        mock_embeddings = Mock()
        mock_embeddings.embed.return_value = [0.1] * 384

        id_to_movie = {
            "uuid-0": Movie("Movie 1", "2020", "Action", "Overview 1", "Director 1", "8.0"),
            "uuid-1": Movie("Movie 2", "2021", "Drama", "Overview 2", "Director 2", "7.5"),
        }

        search_similar_movies(
            mock_client, mock_embeddings, "test query",
            id_to_movie, top_k=5
        )

        captured = capsys.readouterr()
        assert "test query" in captured.out
        assert "0.95" in captured.out
        assert "Movie 1" in captured.out

    def test_search_with_genre_filter(self, capsys):
        """Test searching with genre filter."""
        from vectorpp import SearchResult

        mock_client = Mock()
        mock_client.search.return_value = [
            SearchResult(id="uuid-0", score=0.90, metadata="Action"),
        ]

        mock_embeddings = Mock()
        mock_embeddings.embed.return_value = [0.1] * 384

        id_to_movie = {
            "uuid-0": Movie("Action Movie", "2020", "Action", "Explosions!", "Director", "8.0"),
        }

        search_similar_movies(
            mock_client, mock_embeddings, "action movie",
            id_to_movie, top_k=5, filter_genre="Action"
        )

        captured = capsys.readouterr()
        assert "Action" in captured.out
        mock_client.search.assert_called_once()
        # Verify filter was passed
        call_args = mock_client.search.call_args
        assert call_args[1]['filter_metadata'] == "Action"


class TestInteractiveMode:
    """Tests for the interactive search mode."""

    def test_interactive_mode_search_query(self, capsys):
        """Test that interactive mode performs searches correctly."""
        from vectorpp import SearchResult

        mock_client = Mock()
        mock_client.search.return_value = [
            SearchResult(id="uuid-0", score=0.95, metadata="Action"),
            SearchResult(id="uuid-1", score=0.85, metadata="Sci-Fi"),
        ]

        mock_embeddings = Mock()
        mock_embeddings.embed.return_value = [0.1] * 384

        id_to_movie = {
            "uuid-0": Movie("The Matrix", "1999", "Action, Sci-Fi", "A hacker discovers reality.", "Wachowskis", "8.7"),
            "uuid-1": Movie("Inception", "2010", "Action, Sci-Fi", "Dream within a dream.", "Christopher Nolan", "8.8"),
        }

        # Simulate user input: search query, then quit
        with patch('builtins.input', side_effect=["space adventure", "quit"]):
            interactive_mode(mock_client, mock_embeddings, id_to_movie)

        captured = capsys.readouterr()
        # Should show the interactive mode header
        assert "INTERACTIVE MOVIE SEARCH" in captured.out
        # Should show search results
        assert "space adventure" in captured.out
        assert "The Matrix" in captured.out
        assert "0.95" in captured.out

    def test_interactive_mode_quit_command(self, capsys):
        """Test that quit command exits the interactive mode."""
        mock_client = Mock()
        mock_embeddings = Mock()
        id_to_movie = {}

        # Test various quit commands
        for quit_cmd in ['quit', 'exit', 'q']:
            with patch('builtins.input', side_effect=[quit_cmd]):
                interactive_mode(mock_client, mock_embeddings, id_to_movie)

            captured = capsys.readouterr()
            assert "Goodbye!" in captured.out

    def test_interactive_mode_genre_filter_command(self, capsys):
        """Test setting genre filter in interactive mode."""
        mock_client = Mock()
        mock_embeddings = Mock()
        id_to_movie = {}

        with patch('builtins.input', side_effect=["genre:Action", "quit"]):
            interactive_mode(mock_client, mock_embeddings, id_to_movie)

        captured = capsys.readouterr()
        assert "Genre filter set to: 'Action'" in captured.out

    def test_interactive_mode_top_k_command(self, capsys):
        """Test setting top-k results count in interactive mode."""
        mock_client = Mock()
        mock_embeddings = Mock()
        id_to_movie = {}

        with patch('builtins.input', side_effect=["top:10", "quit"]):
            interactive_mode(mock_client, mock_embeddings, id_to_movie)

        captured = capsys.readouterr()
        assert "Results count set to: 10" in captured.out

    def test_interactive_mode_invalid_top_k(self, capsys):
        """Test handling invalid top-k value."""
        mock_client = Mock()
        mock_embeddings = Mock()
        id_to_movie = {}

        with patch('builtins.input', side_effect=["top:invalid", "quit"]):
            interactive_mode(mock_client, mock_embeddings, id_to_movie)

        captured = capsys.readouterr()
        assert "Invalid number" in captured.out

    def test_interactive_mode_empty_input_ignored(self, capsys):
        """Test that empty input is ignored."""
        mock_client = Mock()
        mock_embeddings = Mock()
        id_to_movie = {}

        with patch('builtins.input', side_effect=["", "quit"]):
            interactive_mode(mock_client, mock_embeddings, id_to_movie)

        # Should not crash and exit cleanly
        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    def test_interactive_mode_keyboard_interrupt(self, capsys):
        """Test handling keyboard interrupt (Ctrl+C)."""
        mock_client = Mock()
        mock_embeddings = Mock()
        id_to_movie = {}

        with patch('builtins.input', side_effect=KeyboardInterrupt()):
            interactive_mode(mock_client, mock_embeddings, id_to_movie)

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    def test_interactive_mode_eof(self, capsys):
        """Test handling EOF (Ctrl+D)."""
        mock_client = Mock()
        mock_embeddings = Mock()
        id_to_movie = {}

        with patch('builtins.input', side_effect=EOFError()):
            interactive_mode(mock_client, mock_embeddings, id_to_movie)

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    def test_interactive_mode_shows_top_5_by_default(self, capsys):
        """Test that interactive mode shows top 5 results by default."""
        from vectorpp import SearchResult

        mock_client = Mock()
        mock_client.search.return_value = [
            SearchResult(id=f"uuid-{i}", score=0.9 - i * 0.1, metadata="Action")
            for i in range(5)
        ]

        mock_embeddings = Mock()
        mock_embeddings.embed.return_value = [0.1] * 384

        id_to_movie = {
            f"uuid-{i}": Movie(f"Movie {i}", "2020", "Action", f"Overview {i}", "Director", "8.0")
            for i in range(5)
        }

        with patch('builtins.input', side_effect=["action movies", "quit"]):
            interactive_mode(mock_client, mock_embeddings, id_to_movie)

        # Verify search was called with k=5 (default)
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args[1]['k'] == 5

    def test_interactive_mode_displays_timing(self, capsys):
        """Test that interactive mode displays response time."""
        from vectorpp import SearchResult

        mock_client = Mock()
        mock_client.search.return_value = [
            SearchResult(id="uuid-0", score=0.95, metadata="Action"),
        ]

        mock_embeddings = Mock()
        mock_embeddings.embed.return_value = [0.1] * 384

        id_to_movie = {
            "uuid-0": Movie("Test Movie", "2020", "Action", "Overview", "Director", "8.0"),
        }

        with patch('builtins.input', side_effect=["test", "quit"]):
            interactive_mode(mock_client, mock_embeddings, id_to_movie)

        captured = capsys.readouterr()
        # Should display timing info (Embedding, Search, Total)
        assert "Embedding:" in captured.out
        assert "Search:" in captured.out
        assert "Total:" in captured.out
        assert "ms" in captured.out
