TRANSLATIONS = {

    "en": {

        # --- Header ---
        "title": "Public Consultation Analysis",
        "subtitle": "Rule-based transparency tool for consultation analysis.",

        # --- Input ---
        "input_label": "Enter consultation URL",
        "run": "Run Analysis",

        # --- Advanced ---
        "advanced": "Advanced Settings",
        "policy": "Policy keywords (comma separated)",
        "amend": "Amendment verbs (comma separated)",

        # --- Runtime ---
        "scraping": "Scraping and collecting comments...",
        "scraping_chapter": "Scraping chapter",
        "loaded_cache": "Results loaded from cache.",
        "completed": "Analysis completed.",
        "abort": "Abort",
        "no_comments": "No comments found or scraping aborted.",

        # --- Duplicate Detection ---
        "duplicate_method_label": "Duplicate detection method",
        "duplicate_templates": "Duplicate Templates",
        "exact_match": "Exact match",
        "fuzzy_match": "Fuzzy match",
        "similarity_threshold": "Similarity threshold (%)",

        # --- Core Metrics ---
        "campaign": "Campaign Share (%)",
        "strict": "Strict Legislative Layer (%)",
        "mean": "Mean words",
        "median": "Median words",
        "std": "Std deviation",
        "max": "Max words",
        "total": "Total Comments",

        # --- Tooltips ---
        "campaign_help": (
            "Percentage of comments that are textual duplicates of other submissions. "
            "Approximates organized or template-based participation."
        ),
        "strict_desc": (
            "Percentage of comments containing both an article reference "
            "and an explicit amendment proposal."
        ),
        "mean_help": "Arithmetic average number of words per comment.",
        "median_help": "Middle value of word-count distribution.",
        "std_help": "Standard deviation of comment length.",

        # --- Distribution ---
        "distribution": "Comment Length Distribution (Kernel Density Estimation)",
        "density": "Density",
        "mean_line": "Mean",
        "median_line": "Median",
        "word_count_label": "Word count",

        # --- Templates ---
        "top_templates": "Top Duplicate Templates",
        "occurrences": "Occurrences",
        "show_full_text": "Show full text",

        # --- Method Panel ---
        "method_panel": "Methodological Parameters",
        "execution_summary": "Execution Summary",
        "active_configuration": "Active Configuration",
        "timestamp": "Run timestamp",

        # --- Footer / Disclaimer ---
        "disclaimer_text": (
        "This application is a work in progress developed within the "
        "Postgraduate Programme «e-Government» of the University of the Aegean. "
        "It aims to extract and analyze comments submitted to public consultations "
        "on opengov.gr."
        ),
    
        "methodology_note": (
        "The analysis is fully rule-based and methodologically transparent. "
        "Metrics are calculated using predefined textual patterns and similarity thresholds, editable by users."
        ),
        "developed_by": "Developed with ❤️ by",

        "code_available": (
        "The source code is openly available at "
        "<a href='https://github.com/rinenweb/consultation-analyzer/' target='_blank'>GitHub</a>."
        ),

        "chapter_table_headers": [
            "Chapter ID",
            "Chapter Title",
            "Comment Count"
        ]
    },

    # ======================================================

    "el": {

        # --- Header ---
        "title": "Ανάλυση Δημόσιας Διαβούλευσης",
        "subtitle": "Εργαλείο διαφάνειας βασισμένο σε κανόνες.",

        # --- Input ---
        "input_label": "Εισάγετε URL διαβούλευσης από το opengov.gr",
        "run": "Εκτέλεση Ανάλυσης",

        # --- Advanced ---
        "advanced": "Προχωρημένες Ρυθμίσεις",
        "policy": "Λέξεις πολιτικής (διαχωρισμένες με κόμμα)",
        "amend": "Ρήματα τροποποίησης (διαχωρισμένα με κόμμα)",

        # --- Runtime ---
        "scraping": "Συλλογή και καταγραφή σχολίων...",
        "scraping_chapter": "Συλλογή κεφαλαίου",
        "loaded_cache": "Τα αποτελέσματα φορτώθηκαν από cache.",
        "completed": "Η ανάλυση ολοκληρώθηκε.",
        "abort": "Ακύρωση",
        "no_comments": "Δεν βρέθηκαν σχόλια ή η διαδικασία ακυρώθηκε.",

        # --- Duplicate Detection ---
        "duplicate_method_label": "Μέθοδος εντοπισμού διπλότυπων",
        "duplicate_templates": "Μοτίβα Διπλότυπων Σχολίων",
        "exact_match": "Ακριβής Ταύτιση",
        "fuzzy_match": "Προσεγγιστική Ταύτιση",
        "similarity_threshold": "Κατώφλι Ομοιότητας (%)",

        # --- Core Metrics ---
        "campaign": "Ποσοστό Campaign (%)",
        "strict": "Στρώμα Στοχευμένης Νομοθετικής Παρέμβασης (%)",
        "mean": "Μέσος Όρος Λέξεων",
        "median": "Διάμεσος",
        "std": "Τυπική Απόκλιση",
        "max": "Μέγιστο",
        "total": "Σύνολο Σχολίων",

        # --- Tooltips ---
        "campaign_help": (
            "Ποσοστό σχολίων που αποτελούν κειμενικά αντίγραφα άλλων υποβολών. "
            "Προσεγγίζει οργανωμένη συμμετοχή τύπου template."
        ),
        "strict_desc": (
            "Ποσοστό σχολίων που περιέχουν ταυτόχρονα αναφορά σε άρθρο "
            "και ρητή πρόταση τροποποίησης."
        ),
        "mean_help": "Αριθμητικός μέσος όρος λέξεων ανά σχόλιο.",
        "median_help": "Κεντρική τιμή της κατανομής μήκους.",
        "std_help": "Τυπική απόκλιση μήκους σχολίων.",

        # --- Distribution ---
        "distribution": "Κατανομή Μήκους Σχολίων (Εκτίμηση Πυκνότητας Kernel)",
        "density": "Πυκνότητα",
        "mean_line": "Μέσος Όρος",
        "median_line": "Διάμεσος",
        "word_count_label": "Αριθμός λέξεων",

        # --- Templates ---
        "top_templates": "Κορυφαία Επαναλαμβανόμενα Templates",
        "occurrences": "Εμφανίσεις",
        "show_full_text": "Εμφάνιση πλήρους κειμένου",

        # --- Method Panel ---
        "method_panel": "Μεθοδολογικές Παράμετροι",
        "execution_summary": "Σύνοψη Εκτέλεσης",
        "active_configuration": "Ενεργές Ρυθμίσεις",
        "timestamp": "Χρόνος Εκτέλεσης",

        # --- Footer / Disclaimer ---
        "disclaimer_text": (
            "Το παρόν εργαλείο αποτελεί έργο εν εξελίξει στο πλαίσιο του "
            "Μεταπτυχιακού Προγράμματος «Ηλεκτρονική Διακυβέρνηση» "
            "του Πανεπιστημίου Αιγαίου και αποσκοπεί στην εξαγωγή και "
            "ανάλυση σχολίων που υποβάλλονται σε δημόσιες διαβουλεύσεις "
            "στο opengov.gr."
        ),
        
        "methodology_note": (
            "Η ανάλυση είναι πλήρως βασισμένη σε κανόνες και μεθοδολογικά διαφανής. "
            "Οι μετρικές υπολογίζονται βάσει προκαθορισμένων γλωσσικών προτύπων "
            "και κατωφλιών ομοιότητας, τα οποία είναι διαθέσιμα προς επεξεργασία από τους χρήστες."
        ),
        
        "developed_by": "Developed with ❤️ by",

        "code_available": (
        "Ο πηγαίος κώδικας είναι δημόσια διαθέσιμος στο "
        "<a href='https://github.com/rinenweb/consultation-analyzer/' target='_blank'>GitHub</a>."
        ),

        "chapter_table_headers": [
            "ID Κεφαλαίου",
            "Τίτλος Κεφαλαίου",
            "Αριθμός Σχολίων"
        ]
    }
}
