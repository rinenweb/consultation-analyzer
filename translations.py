TRANSLATIONS = {
    "en": {
        # --- Header ---
        "title": "Public Consultation Analysis",
        "subtitle": "Rule-based transparency tool for consultation analysis.",
        # --- Input ---
        "input_label": "Enter consultation URL or parent ID",
        "run": "Run Analysis",
        # --- Advanced ---
        "advanced": "Advanced Settings",
        "stopwords": "Stopwords (comma separated)",
        "policy": "Policy keywords (comma separated)",
        "amend": "Amendment verbs (comma separated)",
        # --- Runtime ---
        "scraping": "Scraping and analyzing consultation...",
        "completed": "Analysis completed.",
        # --- Core Metrics ---
        "total": "Total Comments",
        "campaign": "Campaign Share (%)",
        "templates": "Duplicate Templates",
        # --- Tooltips ---
        "campaign_help": (
            "Percentage of comments that are exact textual duplicates. "
            "This metric approximates organized or template-based participation."
        ),
        "templates_help": (
            "Number of distinct comment texts that were submitted more than once. "
            "Each unique repeated text is counted once."
        ),
        # --- Statistics ---
        "stats": "Text Statistics",
        "mean": "Mean words",
        "median": "Median words",
        "max": "Max words",
        "std": "Std deviation",
        "mean_help": (
            "Arithmetic average number of words per comment."
        ),
        "median_help": (
            "The middle value of the word-count distribution. "
            "Less sensitive to extreme values than the mean."
        ),
        "std_help": (
            "Standard deviation of comment length. "
            "Indicates dispersion around the mean."
        ),
        # --- Distribution ---
        "distribution": "Comment Length Distribution (Kernel Density Estimation)",
        # --- Strict Layer ---
        "strict": "Strict Legislative Layer",
        "strict_desc": (
            "Percentage of comments containing both an article reference "
            "and an explicit amendment proposal."
        ),
        # --- Methodological Panel ---
        "method_panel": "Methodological Parameters & Execution Transparency",
        "execution_summary": "Execution Summary",
        "metric_definitions": "Metric Definitions",
        "definitions_text": (
            "**Campaign Share** measures the proportion of comments that are "
            "exact duplicates of other submissions.\n\n"
            "**Duplicate Templates** counts how many distinct texts were "
            "submitted multiple times.\n\n"
            "**Strict Legislative Layer** estimates the share of comments "
            "that simultaneously reference a legislative article and propose "
            "a textual amendment."
        ),
         # --- Duplicate Detection ---
        "duplicate_method_label": "Duplicate detection method",
        "exact_match": "Exact match",
        "fuzzy_match": "Fuzzy match",
        "similarity_threshold": "Similarity threshold (%)",

        # --- Top Templates ---
        "top_templates": "Top Duplicate Templates",
        "occurrences": "Occurrences",
        "show_full_text": "Show full text",

        # --- KDE Labels ---
        "density": "Density",
        "mean_line": "Mean",
        "median_line": "Median",
        "word_count_label": "Word count",

        "active_configuration": "Active Configuration",
        "chapter_table_headers": [
            "Chapter ID",
            "Chapter Title",
            "Comment Count"
        ],
        "timestamp": "Run timestamp"
    },

    # =========================================================
    # ======================= GREEK ===========================
    # =========================================================

    "el": {

        # --- Header ---
        "title": "Ανάλυση Δημόσιας Διαβούλευσης",
        "subtitle": "Εργαλείο διαφάνειας βασισμένο σε κανόνες.",

        # --- Input ---
        "input_label": "Εισάγετε URL ή ID διαβούλευσης από το opengov.gr",
        "run": "Εκτέλεση Ανάλυσης",

        # --- Advanced ---
        "advanced": "Προχωρημένες Ρυθμίσεις",
        "stopwords": "Stopwords (διαχωρισμένα με κόμμα)",
        "policy": "Λέξεις πολιτικής (διαχωρισμένες με κόμμα)",
        "amend": "Ρήματα τροποποίησης (διαχωρισμένα με κόμμα)",

        # --- Runtime ---
        "scraping": "Συλλογή και ανάλυση σχολίων...",
        "completed": "Η ανάλυση ολοκληρώθηκε.",

        # --- Core Metrics ---
        "total": "Σύνολο Σχολίων",
        "campaign": "Ποσοστό Campaign (%)",
        "templates": "Διαφορετικά Templates",

        # --- Tooltips ---
        "campaign_help": (
            "Ποσοστό σχολίων που αποτελούν ακριβή κειμενικά αντίγραφα "
            "άλλων υποβολών. Προσεγγίζει οργανωμένη ή template-based συμμετοχή."
        ),

        "templates_help": (
            "Αριθμός διαφορετικών κειμένων που υποβλήθηκαν περισσότερες από μία φορές. "
            "Κάθε μοναδικό επαναλαμβανόμενο κείμενο μετράται μία φορά."
        ),

        # --- Statistics ---
        "stats": "Στατιστικά Κειμένου",
        "mean": "Μέσος Όρος Λέξεων",
        "median": "Διάμεσος",
        "max": "Μέγιστο",
        "std": "Τυπική Απόκλιση",

        "mean_help": (
            "Αριθμητικός μέσος όρος λέξεων ανά σχόλιο."
        ),

        "median_help": (
            "Η κεντρική τιμή της κατανομής μήκους σχολίων. "
            "Είναι λιγότερο ευαίσθητη σε ακραίες τιμές."
        ),

        "std_help": (
            "Τυπική απόκλιση του μήκους σχολίων. "
            "Δείχνει τον βαθμό διασποράς γύρω από τον μέσο όρο."
        ),

        # --- Distribution ---
        "distribution": "Κατανομή Μήκους Σχολίων (Εκτίμηση Πυκνότητας Kernel)",

        # --- Strict Layer ---
        "strict": "Στρώμα Στοχευμένης Νομοθετικής Παρέμβασης",
        "strict_desc": (
            "Ποσοστό σχολίων που περιέχουν ταυτόχρονα αναφορά σε άρθρο "
            "και σαφή πρόταση τροποποίησης."
        ),

        # --- Methodological Panel ---
        "method_panel": "Μεθοδολογικές Παράμετροι & Διαφάνεια Εκτέλεσης",

        "execution_summary": "Σύνοψη Εκτέλεσης",

        "metric_definitions": "Ορισμοί Μετρικών",

        "definitions_text": (
            "**Ποσοστό Campaign**: Ποσοστό σχολίων που αποτελούν ακριβή "
            "αντίγραφα άλλων υποβολών.\n\n"
            "**Διαφορετικά Templates**: Πλήθος μοναδικών κειμένων που "
            "υποβλήθηκαν επανειλημμένα.\n\n"
            "**Στρώμα Στοχευμένης Νομοθετικής Παρέμβασης**: "
            "Εκτιμά το ποσοστό σχολίων που αναφέρονται σε συγκεκριμένο άρθρο "
            "και ταυτόχρονα προτείνουν ρητή τροποποίηση."
        ),
        # --- Duplicate Detection ---
        "duplicate_method_label": "Μέθοδος εντοπισμού διπλότυπων",
        "exact_match": "Ακριβής Ταύτιση",
        "fuzzy_match": "Προσεγγιστική Ταύτιση",
        "similarity_threshold": "Κατώφλι Ομοιότητας (%)",

        # --- Top Templates ---
        "top_templates": "Κορυφαία Επαναλαμβανόμενα Templates",
        "occurrences": "Εμφανίσεις",
        "show_full_text": "Εμφάνιση πλήρους κειμένου",

        # --- KDE Labels ---
        "density": "Πυκνότητα",
        "mean_line": "Μέσος Όρος",
        "median_line": "Διάμεσος",
        "word_count_label": "Αριθμός λέξεων",

        "active_configuration": "Ενεργές Ρυθμίσεις",

        "chapter_table_headers": [
            "ID Κεφαλαίου",
            "Τίτλος Κεφαλαίου",
            "Αριθμός Σχολίων"
        ],

        "timestamp": "Χρόνος Εκτέλεσης"
    }
}
