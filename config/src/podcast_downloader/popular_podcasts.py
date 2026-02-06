"""database of popular us podcasts for enhanced search functionality."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PopularPodcast:
    """information about a popular podcast."""

    rank: int
    title: str
    host: str
    genre: str
    description: str
    rss_feed: Optional[str] = None
    search_aliases: list[str] = field(default_factory=list)


# top 50 us podcasts database (based on edison research + apple podcasts charts)
TOP_US_PODCASTS: list[PopularPodcast] = [
    PopularPodcast(
        1,
        "The Joe Rogan Experience",
        "Joe Rogan",
        "Comedy",
        "Long-form conversations with guests from various fields",
        "https://feeds.megaphone.fm/GLT1412515089",
        ["joe rogan", "rogan", "jre"],
    ),
    PopularPodcast(
        2,
        "Crime Junkie",
        "Ashley Flowers & Brit Prawat",
        "True Crime",
        "Weekly true crime podcast covering unsolved cases",
        "https://feeds.megaphone.fm/ADL9840290619",
        ["crime junkie", "ashley flowers"],
    ),
    PopularPodcast(
        3,
        "The Daily",
        "The New York Times",
        "News",
        "Daily news podcast from The New York Times",
        "https://feeds.simplecast.com/54nAGcIl",
        ["daily", "nyt daily", "new york times", "nyt"],
    ),
    PopularPodcast(
        4,
        "Call Her Daddy",
        "Alex Cooper",
        "Comedy",
        "Unfiltered conversations about relationships and pop culture",
        "https://feeds.megaphone.fm/ADL6868840599",
        ["call her daddy", "alex cooper"],
    ),
    PopularPodcast(
        5,
        "This Past Weekend w/ Theo Von",
        "Theo Von",
        "Comedy",
        "Comedian Theo Von's take on current events and life",
        "https://feeds.megaphone.fm/CAD8812407396",
        ["theo von", "this past weekend"],
    ),
    PopularPodcast(
        6,
        "Dateline NBC",
        "NBC News",
        "True Crime",
        "In-depth true crime investigations from NBC",
        "https://podcastfeeds.nbcnews.com/dateline-nbc",
        ["dateline", "dateline nbc"],
    ),
    PopularPodcast(
        7,
        "SmartLess",
        "Jason Bateman, Sean Hayes, Will Arnett",
        "Comedy",
        "Three friends have unscripted conversations with mystery guests",
        "https://feeds.megaphone.fm/smartless",
        ["smartless"],
    ),
    PopularPodcast(
        8,
        "Huberman Lab",
        "Andrew Huberman",
        "Health",
        "Neuroscience and health optimization protocols",
        "https://feeds.megaphone.fm/hubermanlab",
        ["huberman", "huberman lab", "andrew huberman"],
    ),
    PopularPodcast(
        9,
        "The Ben Shapiro Show",
        "Ben Shapiro",
        "News",
        "Daily political analysis and conservative commentary",
        "https://feeds.megaphone.fm/WWO3519750118",
        ["ben shapiro", "shapiro"],
    ),
    PopularPodcast(
        10,
        "Serial",
        "Serial Productions",
        "True Crime",
        "Investigative journalism told week by week",
        "https://feeds.simplecast.com/xl36XBC2",
        ["serial"],
    ),
    PopularPodcast(
        11,
        "Stuff You Should Know",
        "iHeartPodcasts",
        "Education",
        "Educational podcast covering how things work",
        "https://feeds.megaphone.fm/stuffyoushouldknow",
        ["stuff you should know", "sysk"],
    ),
    PopularPodcast(
        12,
        "My Favorite Murder",
        "Karen Kilgariff & Georgia Hardstark",
        "True Crime",
        "True crime comedy podcast",
        "https://rss.art19.com/my-favorite-murder",
        ["my favorite murder", "mfm"],
    ),
    PopularPodcast(
        13,
        "The Ezra Klein Show",
        "The New York Times",
        "News",
        "In-depth conversations about ideas, politics and culture",
        "https://feeds.simplecast.com/82FI35Px",
        ["ezra klein", "ezra klein show"],
    ),
    PopularPodcast(
        14,
        "Up First",
        "NPR",
        "News",
        "Daily morning news briefing from NPR",
        "https://feeds.npr.org/510318/podcast.xml",
        ["up first", "npr up first"],
    ),
    PopularPodcast(
        15,
        "Radiolab",
        "WNYC Studios",
        "Science",
        "Investigative journalism about science and philosophy",
        "https://feeds.wnyc.org/radiolab",
        ["radiolab"],
    ),
    PopularPodcast(
        16,
        "This American Life",
        "Ira Glass",
        "Society",
        "Weekly public radio show and podcast",
        "https://www.thisamericanlife.org/podcast/rss.xml",
        ["this american life", "tal", "ira glass"],
    ),
    PopularPodcast(
        17,
        "Freakonomics Radio",
        "Stephen Dubner",
        "Business",
        "Economics and human behavior exploration",
        "https://feeds.simplecast.com/Y8lFbOT4",
        ["freakonomics", "freakonomics radio"],
    ),
    PopularPodcast(
        18,
        "Pod Save America",
        "Crooked Media",
        "News",
        "Political podcast from former Obama staffers",
        "https://feeds.megaphone.fm/PSA8297440886",
        ["pod save america", "psa"],
    ),
    PopularPodcast(
        19,
        "Hidden Brain",
        "NPR",
        "Science",
        "Unconscious patterns in human behavior",
        "https://feeds.npr.org/510308/podcast.xml",
        ["hidden brain"],
    ),
    PopularPodcast(
        20,
        "Fresh Air",
        "NPR / Terry Gross",
        "Interview",
        "Interviews with cultural tastemakers",
        "https://feeds.npr.org/381444908/podcast.xml",
        ["fresh air", "terry gross"],
    ),
    PopularPodcast(
        21,
        "Conan O'Brien Needs A Friend",
        "Conan O'Brien",
        "Comedy",
        "Conan interviews celebrity guests and friends",
        "https://feeds.simplecast.com/dHoohVNH",
        ["conan", "conan o'brien", "conan needs a friend"],
    ),
    PopularPodcast(
        22,
        "Lex Fridman Podcast",
        "Lex Fridman",
        "Science",
        "Deep conversations about AI, science, and philosophy",
        "https://lexfridman.com/feed/podcast/",
        ["lex fridman", "lex"],
    ),
    PopularPodcast(
        23,
        "The Tim Ferriss Show",
        "Tim Ferriss",
        "Business",
        "Deconstructing world-class performers",
        "https://rss.art19.com/tim-ferriss-show",
        ["tim ferriss", "ferriss"],
    ),
    PopularPodcast(
        24,
        "Armchair Expert with Dax Shepard",
        "Dax Shepard",
        "Interview",
        "Celebrity interviews exploring human experiences",
        "https://feeds.megaphone.fm/armchair-expert",
        ["armchair expert", "dax shepard"],
    ),
    PopularPodcast(
        25,
        "How I Built This with Guy Raz",
        "NPR / Guy Raz",
        "Business",
        "Stories behind successful companies and movements",
        "https://feeds.npr.org/510313/podcast.xml",
        ["how i built this", "guy raz"],
    ),
    PopularPodcast(
        26,
        "Planet Money",
        "NPR",
        "Business",
        "Economics made fun and accessible",
        "https://feeds.npr.org/510289/podcast.xml",
        ["planet money"],
    ),
    PopularPodcast(
        27,
        "Wait Wait... Don't Tell Me!",
        "NPR / Peter Sagal",
        "Comedy",
        "NPR's weekly hour-long quiz program",
        "https://feeds.npr.org/344098539/podcast.xml",
        ["wait wait", "dont tell me"],
    ),
    PopularPodcast(
        28,
        "TED Talks Daily",
        "TED",
        "Education",
        "Daily TED talks on ideas worth spreading",
        "https://feeds.feedburner.com/TEDTalks_audio",
        ["ted talks", "ted daily"],
    ),
    PopularPodcast(
        29,
        "The Moth",
        "The Moth",
        "Society",
        "True stories told live onstage",
        "https://feeds.megaphone.fm/themoth",
        ["the moth", "moth"],
    ),
    PopularPodcast(
        30,
        "Revisionist History",
        "Malcolm Gladwell",
        "Society",
        "Reexamining overlooked or misunderstood events",
        "https://feeds.megaphone.fm/revisionisthistory",
        ["revisionist history", "malcolm gladwell"],
    ),
    PopularPodcast(
        31,
        "99% Invisible",
        "Roman Mars",
        "Design",
        "Design and architecture stories",
        "https://feeds.simplecast.com/BqbsxVfO",
        ["99 invisible", "roman mars"],
    ),
    PopularPodcast(
        32,
        "Hardcore History",
        "Dan Carlin",
        "History",
        "Epic deep-dives into historical topics",
        "https://feeds.feedburner.com/dancarlonaudio",
        ["hardcore history", "dan carlin"],
    ),
    PopularPodcast(
        33,
        "Office Ladies",
        "Jenna Fischer & Angela Kinsey",
        "Comedy",
        "The Office rewatch podcast with behind-the-scenes stories",
        "https://feeds.megaphone.fm/officeladies",
        ["office ladies", "the office"],
    ),
    PopularPodcast(
        34,
        "Casefile True Crime",
        "Anonymous Host",
        "True Crime",
        "Detailed true crime investigations from Australia",
        "https://audioboom.com/channels/4322676.rss",
        ["casefile", "casefile true crime"],
    ),
    PopularPodcast(
        35,
        "The Indicator from Planet Money",
        "NPR",
        "Business",
        "Daily economics stories in under 10 minutes",
        "https://feeds.npr.org/510325/podcast.xml",
        ["the indicator", "indicator"],
    ),
    PopularPodcast(
        36,
        "Sword and Scale",
        "Mike Boudet",
        "True Crime",
        "In-depth true crime stories",
        None,
        ["sword and scale"],
    ),
    PopularPodcast(
        37,
        "Reply All",
        "Gimlet Media",
        "Technology",
        "Stories about the internet and how it shapes our lives",
        "https://feeds.megaphone.fm/replyall",
        ["reply all"],
    ),
    PopularPodcast(
        38,
        "The Jordan B. Peterson Podcast",
        "Jordan Peterson",
        "Education",
        "Psychology, philosophy, and culture discussions",
        "https://feeds.megaphone.fm/WWO6832693779",
        ["jordan peterson", "peterson"],
    ),
    PopularPodcast(
        39,
        "WTF with Marc Maron",
        "Marc Maron",
        "Comedy",
        "Comedians, actors and musicians in conversation",
        "https://feeds.megaphone.fm/wtfpod",
        ["wtf", "marc maron"],
    ),
    PopularPodcast(
        40,
        "The Bill Simmons Podcast",
        "Bill Simmons",
        "Sports",
        "Sports and pop culture analysis",
        "https://feeds.megaphone.fm/the-bill-simmons-podcast",
        ["bill simmons"],
    ),
    PopularPodcast(
        41,
        "Making Sense with Sam Harris",
        "Sam Harris",
        "Philosophy",
        "Discussions about the mind, society, and current events",
        "https://wakingup.libsyn.com/rss",
        ["sam harris", "making sense", "waking up"],
    ),
    PopularPodcast(
        42,
        "The Dollop",
        "Dave Anthony & Gareth Reynolds",
        "Comedy",
        "Comedians discuss American history",
        "https://thedollop.libsyn.com/rss",
        ["the dollop", "dollop"],
    ),
    PopularPodcast(
        43,
        "Ear Hustle",
        "Nigel Poor & Earlonne Woods",
        "Society",
        "Stories of life inside San Quentin State Prison",
        "https://feeds.megaphone.fm/earhustle",
        ["ear hustle"],
    ),
    PopularPodcast(
        44,
        "The Knowledge Project",
        "Shane Parrish",
        "Education",
        "Mastering the best of what others have figured out",
        "https://theknowledgeproject.libsyn.com/rss",
        ["knowledge project", "shane parrish", "farnam street"],
    ),
    PopularPodcast(
        45,
        "No Such Thing As A Fish",
        "QI Elves",
        "Comedy",
        "Weird and wonderful facts from the QI researchers",
        "https://audioboom.com/channels/2399216.rss",
        ["no such thing as a fish", "fish"],
    ),
    PopularPodcast(
        46,
        "StartUp Podcast",
        "Gimlet Media",
        "Business",
        "Stories about what it's like to start a business",
        "https://feeds.megaphone.fm/startup",
        ["startup", "startup podcast"],
    ),
    PopularPodcast(
        47,
        "You're Wrong About",
        "Sarah Marshall",
        "History",
        "Misremembered and misrepresented events",
        "https://feeds.buzzsprout.com/1112270.rss",
        ["you're wrong about", "youre wrong about"],
    ),
    PopularPodcast(
        48,
        "The Daily Show: Ears Edition",
        "Comedy Central",
        "Comedy",
        "The Daily Show in podcast form",
        "https://feeds.simplecast.com/C6NQglnL",
        ["daily show", "trevor noah", "jon stewart"],
    ),
    PopularPodcast(
        49,
        "The Rest Is History",
        "Tom Holland & Dominic Sandbrook",
        "History",
        "Stories that made our world",
        "https://feeds.megaphone.fm/the-rest-is-history",
        ["rest is history", "tom holland"],
    ),
    PopularPodcast(
        50,
        "All-In Podcast",
        "Chamath, Jason, Sacks & Friedberg",
        "Technology",
        "Tech billionaires discuss business and politics",
        "https://feeds.megaphone.fm/all-in-with-chamath-jason-sacks-friedberg",
        ["all in", "all in podcast", "chamath"],
    ),
]


def get_popular_podcasts_by_rank(count: int = 10) -> list[PopularPodcast]:
    """get top n podcasts by rank.

    args:
        count: number of podcasts to return (max 50).

    returns:
        list of popular podcasts sorted by rank.
    """
    return TOP_US_PODCASTS[: min(count, 50)]


def get_podcasts_by_genre(genre: str) -> list[PopularPodcast]:
    """get podcasts filtered by genre.

    args:
        genre: genre name to filter by (case-insensitive).

    returns:
        list of podcasts matching the genre.
    """
    genre_lower = genre.lower()
    return [p for p in TOP_US_PODCASTS if genre_lower in p.genre.lower()]


def get_all_genres() -> list[str]:
    """get all unique genres from the podcast database.

    returns:
        sorted list of unique genre names.
    """
    genres = set()
    for podcast in TOP_US_PODCASTS:
        genres.add(podcast.genre)
    return sorted(genres)


def search_popular_podcasts(query: str) -> list[PopularPodcast]:
    """search popular podcasts by title or aliases.

    args:
        query: search query string.

    returns:
        list of matching podcasts.
    """
    query_lower = query.lower()
    results = []

    for podcast in TOP_US_PODCASTS:
        # check title.
        if query_lower in podcast.title.lower():
            results.append(podcast)
            continue

        # check aliases.
        for alias in podcast.search_aliases:
            if query_lower in alias.lower() or alias.lower() in query_lower:
                results.append(podcast)
                break

    return results


def get_podcast_by_title(title: str) -> Optional[PopularPodcast]:
    """get a podcast by exact title match.

    args:
        title: podcast title to find.

    returns:
        matching podcast or None.
    """
    title_lower = title.lower()
    for podcast in TOP_US_PODCASTS:
        if podcast.title.lower() == title_lower:
            return podcast
    return None


def get_rss_feed_for_popular_podcast(query: str) -> Optional[str]:
    """get rss feed url for a popular podcast if known.

    args:
        query: podcast name or alias to search.

    returns:
        rss feed url if found and available, else None.
    """
    results = search_popular_podcasts(query)
    if results and results[0].rss_feed:
        return results[0].rss_feed
    return None
