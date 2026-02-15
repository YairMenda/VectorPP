#!/usr/bin/env python3
"""Movie Similarity Demo for Vector++.

This demo loads the IMDB Top 1000 movies dataset, generates embeddings
for movie titles and descriptions, inserts them into Vector++, and
provides an interactive search interface to find similar movies.

Prerequisites:
    - Vector++ server running on localhost:50051 (or configure via --host/--port)
    - sentence-transformers package: pip install sentence-transformers
    - pandas package: pip install pandas

Usage:
    # Download the IMDB dataset first (see download_dataset function)
    python movie_demo.py --dataset imdb_top_1000.csv

    # Or generate sample data for testing
    python movie_demo.py --generate-sample

    # Interactive mode (default)
    python movie_demo.py --dataset imdb_top_1000.csv --interactive

    # Search mode (single query)
    python movie_demo.py --dataset imdb_top_1000.csv --query "space adventure"
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent directory to path for vectorpp imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorpp import VectorPPClient, VectorPPError, ConnectionError
from vectorpp.embeddings import HuggingFaceEmbeddings, HuggingFaceError


@dataclass
class Movie:
    """Represents a movie with its metadata."""
    title: str
    year: str
    genre: str
    overview: str
    director: str
    rating: str

    def to_text(self) -> str:
        """Convert movie to text for embedding generation."""
        # Combine title and overview for richer embeddings
        parts = [self.title]
        if self.overview:
            parts.append(self.overview)
        if self.genre:
            parts.append(f"Genre: {self.genre}")
        return " ".join(parts)


def load_movies_from_csv(filepath: str) -> List[Movie]:
    """Load movies from IMDB Top 1000 CSV file.

    Expected CSV columns (flexible matching):
        - Series_Title or Title: Movie title
        - Released_Year or Year: Release year
        - Genre: Movie genres
        - Overview or Description: Plot overview
        - Director: Director name
        - IMDB_Rating or Rating: IMDB rating

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of Movie objects.
    """
    movies = []

    # Column name mappings (lowercase for case-insensitive matching)
    column_mappings = {
        'title': ['series_title', 'title', 'movie_title', 'name'],
        'year': ['released_year', 'year', 'release_year'],
        'genre': ['genre', 'genres'],
        'overview': ['overview', 'description', 'plot', 'synopsis'],
        'director': ['director', 'directors'],
        'rating': ['imdb_rating', 'rating', 'score'],
    }

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        headers_lower = {h.lower(): h for h in reader.fieldnames}

        # Find actual column names
        actual_columns = {}
        for field, possibilities in column_mappings.items():
            for possibility in possibilities:
                if possibility in headers_lower:
                    actual_columns[field] = headers_lower[possibility]
                    break

        if 'title' not in actual_columns:
            raise ValueError(f"Could not find title column in CSV. Headers: {reader.fieldnames}")

        for row in reader:
            try:
                movie = Movie(
                    title=row.get(actual_columns.get('title', ''), '').strip(),
                    year=row.get(actual_columns.get('year', ''), '').strip(),
                    genre=row.get(actual_columns.get('genre', ''), '').strip(),
                    overview=row.get(actual_columns.get('overview', ''), '').strip(),
                    director=row.get(actual_columns.get('director', ''), '').strip(),
                    rating=row.get(actual_columns.get('rating', ''), '').strip(),
                )
                if movie.title:  # Only add movies with titles
                    movies.append(movie)
            except Exception as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue

    return movies


def generate_sample_movies() -> List[Movie]:
    """Generate sample movie data for testing without a CSV file."""
    return [
        Movie("The Shawshank Redemption", "1994", "Drama",
              "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
              "Frank Darabont", "9.3"),
        Movie("The Godfather", "1972", "Crime, Drama",
              "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
              "Francis Ford Coppola", "9.2"),
        Movie("The Dark Knight", "2008", "Action, Crime, Drama",
              "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
              "Christopher Nolan", "9.0"),
        Movie("Inception", "2010", "Action, Adventure, Sci-Fi",
              "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
              "Christopher Nolan", "8.8"),
        Movie("Pulp Fiction", "1994", "Crime, Drama",
              "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
              "Quentin Tarantino", "8.9"),
        Movie("The Matrix", "1999", "Action, Sci-Fi",
              "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
              "The Wachowskis", "8.7"),
        Movie("Interstellar", "2014", "Adventure, Drama, Sci-Fi",
              "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
              "Christopher Nolan", "8.6"),
        Movie("Star Wars: Episode V", "1980", "Action, Adventure, Fantasy",
              "After the Rebels are brutally overpowered by the Empire on the ice planet Hoth, Luke Skywalker begins Jedi training with Yoda.",
              "Irvin Kershner", "8.7"),
        Movie("Forrest Gump", "1994", "Drama, Romance",
              "The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75.",
              "Robert Zemeckis", "8.8"),
        Movie("The Lord of the Rings: The Fellowship of the Ring", "2001", "Action, Adventure, Drama",
              "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron.",
              "Peter Jackson", "8.8"),
        Movie("Fight Club", "1999", "Drama",
              "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into something much, much more.",
              "David Fincher", "8.8"),
        Movie("Goodfellas", "1990", "Biography, Crime, Drama",
              "The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners.",
              "Martin Scorsese", "8.7"),
        Movie("The Silence of the Lambs", "1991", "Crime, Drama, Thriller",
              "A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer.",
              "Jonathan Demme", "8.6"),
        Movie("Schindler's List", "1993", "Biography, Drama, History",
              "In German-occupied Poland during World War II, industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce.",
              "Steven Spielberg", "9.0"),
        Movie("Gladiator", "2000", "Action, Adventure, Drama",
              "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.",
              "Ridley Scott", "8.5"),
        Movie("The Departed", "2006", "Crime, Drama, Thriller",
              "An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston.",
              "Martin Scorsese", "8.5"),
        Movie("Saving Private Ryan", "1998", "Drama, War",
              "Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed in action.",
              "Steven Spielberg", "8.6"),
        Movie("The Prestige", "2006", "Drama, Mystery, Sci-Fi",
              "After a tragic accident, two stage magicians engage in a battle to create the ultimate illusion while sacrificing everything they have to outwit each other.",
              "Christopher Nolan", "8.5"),
        Movie("Spirited Away", "2001", "Animation, Adventure, Family",
              "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits.",
              "Hayao Miyazaki", "8.6"),
        Movie("Whiplash", "2014", "Drama, Music",
              "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential.",
              "Damien Chazelle", "8.5"),
    ]


def print_movie_info(movie: Movie, vector_id: str = None):
    """Print movie information in a formatted way."""
    print(f"  Title: {movie.title} ({movie.year})")
    print(f"  Genre: {movie.genre}")
    print(f"  Director: {movie.director}")
    print(f"  Rating: {movie.rating}")
    if vector_id:
        print(f"  Vector ID: {vector_id}")
    if movie.overview:
        overview = movie.overview[:200] + "..." if len(movie.overview) > 200 else movie.overview
        print(f"  Overview: {overview}")


def load_and_embed_movies(
    client: VectorPPClient,
    movies: List[Movie],
    embeddings: HuggingFaceEmbeddings,
    batch_size: int = 32
) -> Dict[str, Movie]:
    """Load movies into Vector++ with their embeddings.

    Args:
        client: VectorPP client instance.
        movies: List of movies to load.
        embeddings: HuggingFace embeddings instance.
        batch_size: Number of movies to embed at once.

    Returns:
        Dictionary mapping vector IDs to Movie objects.
    """
    id_to_movie = {}
    total = len(movies)

    print(f"\nGenerating embeddings and inserting {total} movies...")
    print(f"Using model: {embeddings.model_name} (dimension: {embeddings.dimensions})")

    start_time = time.time()

    # Process in batches for efficiency
    for i in range(0, total, batch_size):
        batch = movies[i:i + batch_size]
        texts = [m.to_text() for m in batch]

        # Generate embeddings for batch
        try:
            vectors = embeddings.embed_batch(texts)
        except HuggingFaceError as e:
            print(f"Error generating embeddings: {e}")
            continue

        # Insert each movie
        for movie, vector in zip(batch, vectors):
            try:
                # Use genre as metadata for filtering
                vector_id = client.insert(vector, metadata=movie.genre)
                id_to_movie[vector_id] = movie
            except VectorPPError as e:
                print(f"Error inserting '{movie.title}': {e}")
                continue

        # Progress update
        progress = min(i + batch_size, total)
        elapsed = time.time() - start_time
        print(f"  Progress: {progress}/{total} movies ({elapsed:.2f}s)")

    total_time = time.time() - start_time
    print(f"\nLoaded {len(id_to_movie)} movies in {total_time:.2f}s")
    if total_time > 0:
        print(f"Insert rate: {len(id_to_movie) / total_time:.1f} movies/sec")

    return id_to_movie


def search_similar_movies(
    client: VectorPPClient,
    embeddings: HuggingFaceEmbeddings,
    query: str,
    id_to_movie: Dict[str, Movie],
    top_k: int = 5,
    filter_genre: str = ""
) -> None:
    """Search for movies similar to a query.

    Args:
        client: VectorPP client instance.
        embeddings: HuggingFace embeddings instance.
        query: Search query (movie title or description).
        id_to_movie: Mapping of vector IDs to movies.
        top_k: Number of results to return.
        filter_genre: Optional genre filter.
    """
    print(f"\nSearching for: '{query}'")
    if filter_genre:
        print(f"Filtering by genre: {filter_genre}")

    # Generate query embedding
    start_time = time.time()
    try:
        query_vector = embeddings.embed(query)
    except HuggingFaceError as e:
        print(f"Error generating query embedding: {e}")
        return

    embed_time = time.time() - start_time

    # Search Vector++
    search_start = time.time()
    try:
        results = client.search(query_vector, k=top_k, filter_metadata=filter_genre)
    except VectorPPError as e:
        print(f"Error searching: {e}")
        return

    search_time = time.time() - search_start
    total_time = time.time() - start_time

    # Display results
    print(f"\nTop {len(results)} similar movies:")
    print(f"(Embedding: {embed_time*1000:.1f}ms, Search: {search_time*1000:.1f}ms, Total: {total_time*1000:.1f}ms)")
    print("-" * 60)

    for i, result in enumerate(results, 1):
        movie = id_to_movie.get(result.id)
        if movie:
            print(f"\n{i}. Similarity: {result.score:.4f}")
            print_movie_info(movie)
        else:
            print(f"\n{i}. ID: {result.id}, Score: {result.score:.4f}")
            print(f"   Metadata: {result.metadata}")


def interactive_mode(
    client: VectorPPClient,
    embeddings: HuggingFaceEmbeddings,
    id_to_movie: Dict[str, Movie]
) -> None:
    """Run interactive search mode.

    Args:
        client: VectorPP client instance.
        embeddings: HuggingFace embeddings instance.
        id_to_movie: Mapping of vector IDs to movies.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE MOVIE SEARCH")
    print("=" * 60)
    print("Enter a movie title, description, or theme to find similar movies.")
    print("Commands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'genre:<name>' - Filter by genre (e.g., 'genre:Action')")
    print("  'top:<n>' - Change number of results (e.g., 'top:10')")
    print("=" * 60)

    top_k = 5
    filter_genre = ""

    while True:
        try:
            query = input("\nSearch: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        # Handle commands
        if query.lower().startswith('genre:'):
            filter_genre = query[6:].strip()
            print(f"Genre filter set to: '{filter_genre}' (empty to clear)")
            continue

        if query.lower().startswith('top:'):
            try:
                top_k = int(query[4:].strip())
                print(f"Results count set to: {top_k}")
            except ValueError:
                print("Invalid number. Usage: top:5")
            continue

        # Perform search
        search_similar_movies(
            client, embeddings, query, id_to_movie,
            top_k=top_k, filter_genre=filter_genre
        )


def main():
    parser = argparse.ArgumentParser(
        description="Movie Similarity Demo for Vector++",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python movie_demo.py --generate-sample
  python movie_demo.py --dataset imdb_top_1000.csv
  python movie_demo.py --dataset movies.csv --query "space adventure"
  python movie_demo.py --host 192.168.1.100 --port 50051 --generate-sample
        """
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Path to movie CSV file (IMDB Top 1000 format)'
    )
    parser.add_argument(
        '--generate-sample', '-g',
        action='store_true',
        help='Use built-in sample movies (20 movies for testing)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Vector++ server host (default: localhost)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=50051,
        help='Vector++ server port (default: 50051)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model (default: all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single search query (non-interactive mode)'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        default=True,
        help='Run in interactive mode (default: True)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for embedding generation (default: 32)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.dataset and not args.generate_sample:
        print("Error: Must specify either --dataset or --generate-sample")
        parser.print_help()
        sys.exit(1)

    # Load movies
    print("=" * 60)
    print("VECTOR++ MOVIE SIMILARITY DEMO")
    print("=" * 60)

    if args.generate_sample:
        print("\nUsing built-in sample movies...")
        movies = generate_sample_movies()
    else:
        print(f"\nLoading movies from: {args.dataset}")
        if not os.path.exists(args.dataset):
            print(f"Error: File not found: {args.dataset}")
            print("\nTo download the IMDB Top 1000 dataset:")
            print("  1. Go to https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
            print("  2. Download 'imdb_top_1000.csv'")
            print("  3. Run: python movie_demo.py --dataset imdb_top_1000.csv")
            sys.exit(1)
        movies = load_movies_from_csv(args.dataset)

    print(f"Loaded {len(movies)} movies")

    # Initialize embedding model
    print(f"\nInitializing embedding model: {args.model}")
    try:
        embeddings = HuggingFaceEmbeddings(model=args.model)
        # Warm up the model by embedding a test string
        _ = embeddings.embed("test")
        print(f"Model ready (dimension: {embeddings.dimensions})")
    except ImportError:
        print("Error: sentence-transformers package not installed.")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        sys.exit(1)

    # Connect to Vector++ server
    print(f"\nConnecting to Vector++ server at {args.host}:{args.port}...")
    try:
        client = VectorPPClient(host=args.host, port=args.port)
        client.connect()
        print("Connected successfully!")
    except ConnectionError as e:
        print(f"Error: Could not connect to Vector++ server: {e}")
        print("\nMake sure the Vector++ server is running:")
        print("  ./vectorpp_server --config config/default.json")
        sys.exit(1)

    try:
        # Load movies into Vector++
        id_to_movie = load_and_embed_movies(
            client, movies, embeddings,
            batch_size=args.batch_size
        )

        if not id_to_movie:
            print("Error: No movies were loaded. Check the server and dataset.")
            sys.exit(1)

        # Run search
        if args.query:
            # Single query mode
            search_similar_movies(
                client, embeddings, args.query, id_to_movie,
                top_k=args.top_k
            )
        else:
            # Interactive mode
            interactive_mode(client, embeddings, id_to_movie)

    finally:
        client.close()
        print("\nDisconnected from Vector++ server.")


if __name__ == "__main__":
    main()
