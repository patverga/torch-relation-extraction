// Scorer for TAC KBP 2012 slot-filling task
// author:  Ralph Grishman

// version 1.3
// September 17, 2012
// modified by: Hoa Dang (hoa.dang@nist.gov)
//
// Updated to handle revised slot names and response and key formats for 2012.
//
// Each line in the response file has the following tab-separated columns:
//           
// Column 1: query id
// Column 2: slot name
// Column 3: a unique run id for the submission
// Column 4: NIL, if the system believes no information is learnable for this slot; 
//  or a single docid that justifies the relation between the query entity and the slot filler
// Column 5: a slot filler
// Column 6: start offset of filler
// Column 7: end offset of filler
// Column 8: start offset of justification
// Column 9: end offset of justification
// Column 10: confidence score
//
// If Column 4 is NIL, then Columns 5-10 must be empty.
// The slot filler (Column 5) must not contain any embeded tab characters                                                      
//
//
// The slot key file is simply a concatenation of the assessment
// result files from LDC.  Each assessment result files contain 10
// tab-separated fields. The field definitions are as follows:
//
// * item_id       A file-unique integer in the range of 1 to number of
//                 fillers in original LDC assessment file
//           
// * query_name    The query id and slot for the filler; matches the name
//                 of the assessment file for that (query id, slot) pair
// 
// * docid         The document ID for the filler
// 
// * judgment      The judgment of the filler_norm, which will be one of:
//                   -1  - wrong
//                    1  - correct
//                    2  - redundant
//                    3  - inexact 
// 
// * equiv_class   The unique ID of an equivalence class into which this
//                 response falls; zero for incorrect and inexact fillers,
//                 non-zero for correct and redundant fillers.
// 
// * filler_norm   The possibly normalized system-provided filler that was assessed
//
// * judge_filler_raw    The judgment for filler_raw (all unjudged)
//
// * filler_raw    The start and end offsets for the raw filler in docid
//
// * judge_just    The judgment for justification, wich will be one of:
//                   -1 - wrong
//                    1  - correct
//                    3  - inexact 
//
// * justification The start and end offsets for the justification in docid

// version 1.2
// September 16, 2011
// modified by: Hoa Dang (hoa.dang@nist.gov)
//
// Changed format of slot judgment file to match format in assessment
// files from LDC; the slot judgment file is simply a concatenation of
// the assessment result files from LDC.  Each assessment result files
// contain 6 space-separated fields. Space characters after the first
// 5 fields are assumed to be part of the contents of the 6th
// field. The field definitions are as follows:
//
// * item_id       A file-unique integer in the range of 1 to number of
//                 fillers in file
//           
// * query_name    The query id and slot for the filler; matches the name
//                 of the assessment file for that (query id, slot) pair
// 
// * docid         The document ID for the filler
// 
// * judgment      The judgment of the filler, which will be one of:
//                   -1  - wrong
//                    1  - correct
//                    2  - redundant
//                    3  - inexact 
// 
// * equiv_class   The unique ID of an equivalence class into which this
//                 response falls; zero for incorrect and inexact fillers,
//                 non-zero for correct and redundant fillers.
// 
// * filler        The system-provided filler that was assessed


// version 1.1
// September 20, 2010
// modified by: Hoa Dang (hoa.dang@nist.gov)
//
// In trace: distinguish between responses that are redundant (R) with
// reference KB vs responses that are redundant (r) with other
// responses in the run.
//
// Added surprise slots.


// version 1.0
// July 20, 2010
// updated to penalize responses marked REDUNDANT in key
// if slots=... is specified, counts total slots to be filled based on slots file,
//                            rather than response file

// version 0.90
// May 17, 2010

// updated to handle 2010 format responses and keys
// flags added to command line
// take slot list from system response if not separatetly provided

import java.io.*;
import java.util.*;

public class SFScore {

  // true to print out judgement for each line of response
  static boolean trace = false;

  // true to ignore docId ... score only on value
  static boolean anydoc = false;

  // true to ignore case in answers
  static boolean nocase = false;

  // tables built from judgement file

  //  mapping from entity_id:slot_name:response_string:doc_id --> judgement
  static Map<String, Integer> judgement = new HashMap<String, Integer> ();

  //  mapping from entity_id:slot_name:response_string:doc_id --> equivalence class
  static Map<String, Integer> equivalenceClass = new HashMap<String, Integer> ();

  //  mapping from entity_id:slot_name --> {true, false}
  static Map<String, Boolean> query_has_answer = new HashMap<String, Boolean> ();

  //  mapping from entity_id:slot_name --> set of equivalence classes
  static Map<String, Set<Integer>> query_eclasses = new HashMap<String, Set<Integer>> ();

  // table built from response file

  //  mapping from entity_id:slot_name --> list[response_string:doc_id]
  static Map<String, List<String>> response = new HashMap <String, List<String>> ();

  // codes in judgement file
  static final int WRONG = -1;
  static final int CORRECT = 1;
  static final int REDUNDANT = 2;
  static final int INEXACT = 3;

  // next unique equivalence class
  static int eclass_generator = 1000000;

  static String slotFile = null;

  static Set<String> slots = new TreeSet<String>();

  /**
   *  SFScorer <response file> <key file>
   *  scores response file against key file
   */

  public static void main (String[] args) throws IOException {

    if (args.length < 2 || args.length > 6) {
      System.out.println ("SlotScorer must be invoked with 2 to 6 arguments:");
      System.out.println ("\t<response file>  <key file> [flag ...]");
      System.out.println ("flags:");
      System.out.println ("\ttrace  -- print a line with assessment of each system response");
      System.out.println ("\tanydoc -- judge response based only on answer string, ignoring doc id");
      System.out.println ("\tnocase -- ignore case in matching answer string");
      System.out.println ("\tslots=<slotfile> -- take list of entityId:slot pairs from slotfile");
      System.out.println ("\t                    (otherwise list of pairs is taken from system response)");
      System.exit(1);
    }
    String responseFile = args[0];
    String keyFile = args[1];
    for (int i=2; i<args.length; i++) {
      String flag = args[i];
      if (flag.equals("trace")) {
        trace = true;
      } else if (flag.equals("anydoc")) {
        anydoc = true;
      } else if (flag.equals("nocase")) {
        nocase = true;
      } else if (flag.startsWith("slots=")) {
        slotFile = flag.substring(6);
      } else {
        System.out.println ("Unknown flag: " + flag);
        System.exit(1);
      }
    }

    // ----------- read in slot judgements ------------

    BufferedReader keyReader = null;
    try {
      keyReader = new BufferedReader (new FileReader(keyFile));
    } catch (FileNotFoundException e) {
      System.out.println ("Unable to open judgement file " + keyFile);
      System.exit (1);
    }
    String line;
    while ((line = keyReader.readLine()) != null) {
      String[] fields = line.trim().split("\t", 10);
      if (fields.length != 10) {
        System.out.println ("Warning: Invalid line in judgement file:");
        System.out.println (line);
        continue;
      }

      String query_id = fields[1];  //  entity_id + ":" + slot_name
      query_id = query_id.replace(",","/");

      String doc_id = fields[2];
      // 2010 participant annotations may include NILs, but these need not be recorded
      if (doc_id.equals("NIL"))
        continue;
      if (anydoc)
        doc_id = "*";
      String answerString = fields[5];
      answerString = answerString.trim();
      if (nocase)
        answerString = answerString.toLowerCase();
      int jment = 0;
      try {
        jment = Integer.parseInt(fields[3]);
      } catch (NumberFormatException e) {
        System.out.println ("Warning: Invalid line in judgement file -- invalid judgement:");
        System.out.println (line);
        continue;
      }
      int eclass = 0;
      try {
        eclass = Integer.parseInt(fields[4]);
      } catch (NumberFormatException e) {
        System.out.println ("Warning: Invalid line in judgement file -- invalid equivalence class:");
        System.out.println (line);
        continue;
      }
      if (eclass == 0)
        eclass = eclass_generator++;
      String key = query_id + ":" + doc_id + ":" + answerString;
      Integer J = judgement.get(key);
      // For 'anydoc' CORRECT annotations supercede others.
      if (J == null || (anydoc && jment == CORRECT)) {
        judgement.put(key, jment);
        equivalenceClass.put(key, eclass);
        if (jment == CORRECT) {
          query_has_answer.put(query_id, true);
          if (query_eclasses.get(query_id) == null)
            query_eclasses.put(query_id, new HashSet<Integer>());
          query_eclasses.get(query_id).add(eclass);
        }
      } else if (!anydoc){
        if (jment != J) {  // make sure judgments match
          System.out.println("Multiple conflicting judgments for response " + key);
          System.exit (1);
        }
        if (jment == CORRECT) {  // make sure eclasses of CORRECT responses match
          if (equivalenceClass.get(key) != eclass) {
            System.out.println("Multiple equivalence classes for response " + key);
            System.exit (1);
          }
        }
      }
    }
    System.out.println ("Read " + judgement.size() + " judgements.");

    // --------- read in system responses -------------

    BufferedReader responseReader = null;
    try {
      responseReader = new BufferedReader (new FileReader(responseFile));
    } catch (FileNotFoundException e) {
      System.out.println ("Unable to open response file " + responseFile);
      System.exit (1);
    }
    // String line;
    while ((line = responseReader.readLine()) != null) {
      String[] fields = line.trim().split("\t", 10);
      if (fields.length < 4 | fields.length > 10) {
        System.out.println ("Warning: Invalid line in response file:  " + fields.length + " fields");
        System.out.println (line);
        continue;
      }
      String entity = fields[0];
      String slot = fields[1];
      String query_id = entity + ":" + slot;
      String doc_id = fields[3];
      if (anydoc && !doc_id.equals("NIL"))
        doc_id = "*";
      String answer_string = "";
      if (!doc_id.equals("NIL"))
        answer_string = ":" + fields[4].trim();
      if (nocase)
        answer_string = answer_string.toLowerCase();
      if (response.get(query_id) == null)
        response.put(query_id, new ArrayList<String>());
      response.get(query_id).add(doc_id + answer_string);
      slots.add(query_id);
    }
    System.out.println ("Read responses for " + response.size() + " slots.");

    // -------------- read list of slots ----------
    //   separate into single and list valued slots

    if (slotFile != null)
      slots = new TreeSet<String>(readLines(slotFile));
    List<String> svSlots = new ArrayList<String> ();
    List<String> lSlots = new ArrayList<String> ();
    for (String slot : slots) {
      String type = slotType(slot);
      if (type  == "single")
        svSlots.add(slot);
      else if (type == "list")
        lSlots.add(slot);
    }

    // ------------- score responses ------------
    //          for single-valued slots

    // counts for slots with some system response
    int num_sv_slots = 0;
    int num_l_slots = 0;
    // number of non-NIL responses
    int num_responses = 0;
    // number of correct non-NIL responses
    int num_correct = 0;
    // counts for different error types
    int num_wrong = 0;  // includes spurious and incorrect
    int num_inexact = 0;
    int num_redundant = 0;
    // number of correct answers in key 
    //   (correct single-value answers + list-value equivalence classes)
    int num_answers = 0;
    String symbol = "?";

    for (String query : svSlots) {
      if (query_has_answer.get(query) != null)
        num_answers++;
      List<String> responseList = response.get(query);
      if (responseList == null) {
        System.out.println ("Warning: No system response for slot " + query);
        continue;
      }
      num_sv_slots++;
      String responseString = responseList.get(0);
      String fields[] = responseString.split(":",2);
      String doc_id = fields[0];
      String answer_string = "";
      if (fields.length == 2)
        answer_string = fields[1];
      if (doc_id.equals("NIL")) {
        if (query_has_answer.get(query) != null) {
          // missing slot fill
          symbol = "M";
        } else {
          symbol = "C";
        }
      } else /* non-NIL response */ {
        num_responses++;
        Integer J = judgement.get(query + ":" + doc_id + ":" + answer_string);
        if (J == null) {
          System.out.println ("Warning: No judgement for " +
              query + ":" + doc_id + " " + answer_string);
          J = WRONG;
        }
        int j = J;
        switch (j) {
        case WRONG:
          num_wrong++;
          symbol = "W";
          break;
        case REDUNDANT:
          // 			System.out.println
          // 			    ("Single-valued slot tagged 'redundant' in key:" +
          // 			     "\t" + query + ":" + doc_id + " " + answer_string);
          num_redundant++;
          symbol = "R";
          break;
        case CORRECT:
          num_correct++;
          symbol = "C";
          break;
        case INEXACT:
          num_inexact++;
          symbol = "X";
          break;
        default:
          System.out.println ("Warning: Invalid judgement " + j);
        }
      }
      if (trace)
        System.out.println (symbol + " " + query + " " + responseString);
    }

    // ------------- score responses ------------
    //           for list-valued slots

    for (String query : lSlots) {
      int num_answers_to_query = 0;
      if (query_eclasses.get(query) != null)
        num_answers_to_query = query_eclasses.get(query).size();
      num_answers += num_answers_to_query;
      List<String> responseList = response.get(query);
      if (responseList == null) {
        System.out.println ("Warning: No system response for slot " + query);
        continue;
      }
      num_l_slots++;
      Set<Integer> distincts = new HashSet<Integer>();
      int num_responses_to_query = responseList.size();
      for (String responseString : responseList) {
        String fields[] = responseString.split(":",2);
        String doc_id = fields[0];
        String answer_string = "";
        if (fields.length == 2)
          answer_string = fields[1];
        if (doc_id.equals("NIL")) {
          if (num_responses_to_query > 1)
            System.out.println ("Warning: More than one response, including NIL, for " + query);
          num_responses_to_query = 0;
          if (query_has_answer.get(query) != null) {
            // missing system response
            symbol = "M";
          } else {
            symbol = "C";
          }
        } else /* non-NIL system response */ {
          num_responses++;
          String key = query + ":" + doc_id + ":" + answer_string;
          Integer J = judgement.get(key);
          if (J == null) {
            System.out.println ("Warning: No judgement for " + key);
            J = WRONG;
          }
          int j = J;
          switch (j) {
          case WRONG:
            num_wrong++;
            symbol = "W";
            break;
          case REDUNDANT:
            num_redundant++;
            symbol = "R";      // redundant with reference KB
            break;
          case CORRECT:
            Integer E = equivalenceClass.get(key);
            if (distincts.contains(E)) {
              num_redundant++;
              symbol = "r";   // redundant with other returned response
            } else {
              num_correct++;
              symbol = "C";
              distincts.add(E);
            }
            break;
          case INEXACT:
            num_inexact++;
            symbol = "X";
            break;
          default:
            System.out.println ("Warning: Invalid judgement " + j);
          }
        }
        if (trace)
          System.out.println (symbol + " " + query + " " + responseString);
      }
    }
    if (slotFile != null)
      System.out.println ("Slot lists taken from file " + slotFile);
    else
      System.out.println ("Slot lists taken from system response");
    System.out.println ("Slot lists include " + svSlots.size() + " single valued slots");
    System.out.println ("               and " +  lSlots.size() + " list-valued slots");
    System.out.println ("\tNumber of filled slots in key: " + num_answers);
    System.out.println ("\tNumber of filled slots in response: " + num_responses);
    System.out.println ("\tNumber correct non-NIL: " + num_correct);
    System.out.println ("\tNumber redundant: " + num_redundant);
    System.out.println ("\tNumber incorrect / spurious: " + num_wrong);
    System.out.println ("\tNumber inexact: " + num_inexact);

    float recall = ((float) num_correct) / num_answers;
    float precision = ((float) num_correct) / num_responses;
    float F = (2 * recall * precision) / (recall + precision);
    System.out.println ("\nScores:");
    System.out.println ("\tRecall: " + num_correct + " / " + num_answers + " = " + recall);
    System.out.println ("\tPrecision: " + num_correct + " / " + num_responses + " = " + precision);
    System.out.println ("\tF1: " + F);
  }

  /**
   *  reads a series of lines from 'fileName' and returns them as a list of Strings
   */

  static List<String> readLines (String fileName) {
    BufferedReader reader = null;
    List<String> lines = new ArrayList<String>();
    try {
      reader = new BufferedReader (new FileReader(fileName));
    } catch (FileNotFoundException e) {
      System.out.println ("Unable to open file " + fileName);
      System.exit (1);
    }
    String line;
    try {
      while ((line = reader.readLine()) != null) {
        lines.add(line.trim());
      }
    } catch (IOException e) {
      System.out.println ("Error readng from file " + fileName);
      System.exit (1);
    }
    System.out.println ("Read " + lines.size() + " lines from " + fileName);
    return lines;
  }

  static List<String> singleValuedSlots = Arrays.asList(
      "per:date_of_birth",
      "per:age",
      "per:country_of_birth",
      "per:stateorprovince_of_birth",
      "per:city_of_birth",
      "per:date_of_death",
      "per:country_of_death",
      "per:stateorprovince_of_death",
      "per:city_of_death",
      "per:cause_of_death",
      "per:religion",
      "org:number_of_employees_members",
      "org:date_founded",
      "org:date_dissolved",
      "org:country_of_headquarters",
      "org:stateorprovince_of_headquarters",
      "org:city_of_headquarters",
      "org:website");

  static List<String> listValuedSlots = Arrays.asList(
      "per:alternate_names",
      "per:origin",
      "per:countries_of_residence",
      "per:statesorprovinces_of_residence",
      "per:cities_of_residence",
      "per:schools_attended",
      "per:title",
      "per:member_of",
      "per:employee_of",
      "per:employee_or_member_of",
      "per:spouse",
      "per:children",
      "per:parents",
      "per:siblings",
      "per:other_family",
      "per:charges",
      "org:alternate_names",
      "org:political_religious_affiliation",
      "org:top_members_employees",
      "org:members",
      "org:member_of",
      "org:subsidiaries",
      "org:parents",
      "org:founded_by",
      "org:shareholders",
      "per:awards_won",
      "per:charities_supported",
      "per:diseases",
      "org:products");
  /*
   * given entityId:slot, classify slot as "single" or "list" valued
   */

  static String slotType (String slot) {
    String[] slotFields = slot.split(":", 2);
    if (slotFields.length != 2) {
      System.out.println("Invalid slot " + slot);
      return "error";
    }
    if (singleValuedSlots.contains(slotFields[1]))
      return "single";
    if (listValuedSlots.contains(slotFields[1]))
      return "list";
    System.out.println("Invalid slot " + slot);
    // return "list" if you want 2009 slots to be scored too
    return "error"; 
  }
}
