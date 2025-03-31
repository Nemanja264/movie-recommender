function drawPage(host)
{
    const titleDiv = document.createElement('div');
    host.appendChild(titleDiv);

    drawTitle(titleDiv);

    const recommendCont = document.createElement('div');
    recommendCont.className = "recommendation-container";
    host.appendChild(recommendCont);

    drawRecommendationDiv(recommendCont);
}

function drawTitle(titleDiv)
{
    const h1 = document.createElement('h1');
    h1.textContent = "Find Your Next Favorite Movie";
    titleDiv.appendChild(h1)

    const p = document.createElement('p');
    p.textContent = "Discover movies similar to the ones you love.";
    titleDiv.appendChild(p);
}

function drawRecommendationDiv(recommendCont)
{
    const inputMovie = document.createElement('input');
    inputMovie.type = 'text';
    inputMovie.placeholder = "Movie Name";
    inputMovie.className = "movie-input";
    recommendCont.appendChild(inputMovie);

    const btn = document.createElement('button');
    btn.textContent = "Show Recommendations";
    btn.addEventListener('click', recommendMovie);
    recommendCont.appendChild(btn);

    const moviesCont = document.createElement('div');
    moviesCont.className = "movies-container";
    recommendCont.appendChild(moviesCont);
}

function printRecommendations(movies)
{
    const moviesCont = document.querySelector('.movies-container');
    moviesCont.innerHTML = "";

    let p = document.createElement('p');
    p.textContent = "Top movie recommendations:";
    moviesCont.appendChild(p);

    movies.forEach((movie, i) => {
       drawMovie(movie, i, moviesCont);
    });
}

function drawMovie(movie, i, moviesCont)
{
    const p = document.createElement('p');
    p.textContent = `${i+1}. ${movie['title']} (${movie['release_year']})`;

    const imdb_link = movie['imdb_link'];
    if(imdb_link)
    {
        const link = document.createElement('a');
        link.className = "link";
        link.href = imdb_link;
        link.textContent = "[ IMDB ]";
        link.target = "_blank";
        link.rel = "noopener noreferrer";

        p.appendChild(link);
    }

    moviesCont.appendChild(p);
}

async function recommendMovie()
{
    try 
    {
        const movieTitle = document.querySelector('.movie-input').value;
        if(movieTitle.trim() === ""){
            alert("Please enter movie title");
            return;
        }

        const response = await fetch(`/recommend_movie/${movieTitle}`);
        if(!response.ok)
            throw new Error(`Server error: ${response.status}`);

        const result = await response.json();
        console.log(result);

        printRecommendations(result);
    }
    catch (error) 
    {
        console.error("Error fetching data", error);
    }
}

drawPage(document.body);

