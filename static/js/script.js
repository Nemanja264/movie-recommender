function drawPage(host)
{
    drawTitle(host);
    drawRecommendationDiv(host);
}

function drawTitle(host)
{
    const titleDiv = document.createElement('div');
    host.appendChild(titleDiv);

    const h1 = document.createElement('h1');
    h1.textContent = "Find Your Next Favorite Movie";
    titleDiv.appendChild(h1)

    const p = document.createElement('p');
    p.textContent = "Discover movies similar to the ones you love.";
    titleDiv.appendChild(p);
}

function drawRecommendationDiv(host)
{
    const recommendCont = document.createElement('div');
    recommendCont.className = "recommendation-container";
    host.appendChild(recommendCont);

    drawInputDiv(recommendCont);

    const btn = document.createElement('button');
    btn.textContent = "Show Recommendations";
    btn.addEventListener('click', recommendMovie);
    recommendCont.appendChild(btn);

    const moviesCont = document.createElement('div');
    moviesCont.className = "movies-container";
    recommendCont.appendChild(moviesCont);
}

function drawInputDiv(container)
{
    const inputDiv = document.createElement('div');
    inputDiv.className = 'input-container';
    container.appendChild(inputDiv);

    const inputMovie = document.createElement('input');
    inputMovie.type = 'text';
    inputMovie.placeholder = "Movie Name";
    inputMovie.className = "movie-input";
    inputDiv.appendChild(inputMovie);

    const input_numRecs = document.createElement('input');
    input_numRecs.type = 'number';
    input_numRecs.value = 5;
    input_numRecs.placeholder = "Number of recommendations";
    input_numRecs.className = 'numRecs-input';
    inputDiv.appendChild(input_numRecs);
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
        const movieTitle = document.querySelector('.movie-input').value.trim();
        let numRecs = document.querySelector('.numRecs-input').value;
        if(movieTitle === ""){
            alert("Please enter movie title");
            return;
        }

        const response = await fetch(`/recommend_movie/${movieTitle}/${numRecs.toString()}`);
        if(!response.ok)
            throw new Error(`Server error: ${response.status}`);

        const result = await response.json();
        console.log(result);

        if(result)
            printRecommendations(result);
        else
            alert("Please enter another movie name");
    }
    catch (error) 
    {
        console.error("Error fetching data", error);
    }
}

drawPage(document.body);

