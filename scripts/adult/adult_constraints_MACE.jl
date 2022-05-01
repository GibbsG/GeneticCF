plaf = PLAFProgram()

# @GROUP(p, Relationship_cat_0, Relationship_cat_1, Relationship_cat_2, Relationship_cat_3, Relationship_cat_4, Relationship_cat_5)
@GROUP(plaf, s for s in propertynames(X) if contains(string(s), "Relationship"))

# @GROUP(p, Occupation_cat_0, Occupation_cat_1, Occupation_cat_2, Occupation_cat_3, Occupation_cat_4, Occupation_cat_5, Occupation_cat_6, Occupation_cat_7, Occupation_cat_8, Occupation_cat_9, Occupation_cat_10, Occupation_cat_11, Occupation_cat_12, Occupation_cat_13)
@GROUP(plaf, s for s in propertynames(X) if contains(string(s), "Occupation"))

# @GROUP(p, MaritalStatus_cat_0, MaritalStatus_cat_1, MaritalStatus_cat_2, MaritalStatus_cat_3, MaritalStatus_cat_4, MaritalStatus_cat_5, MaritalStatus_cat_6)
@GROUP(plaf, s for s in propertynames(X) if contains(string(s), "MaritalStatus"))

# @GROUP(p, WorkClass_cat_0, WorkClass_cat_1, WorkClass_cat_2, WorkClass_cat_3, WorkClass_cat_4, WorkClass_cat_5, WorkClass_cat_6)
@GROUP(plaf, s for s in propertynames(X) if contains(string(s), "WorkClass"))

@GROUP(plaf, EducationNumber, EducationLevel)